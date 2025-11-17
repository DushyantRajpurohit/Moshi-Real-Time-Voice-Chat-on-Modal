import asyncio

import modal

from .common import app

# Define the path where models will be cached within the Modal Volume
model_path = '/models'
# Define a persistent volume for model caching
volume = modal.Volume.from_name('model_cache', create_if_missing=True)

# Build the base image for the application
image = modal.Image.debian_slim(python_version='3.11')
# Install necessary Python packages
image = image.pip_install('moshi', 'sphn', 'fastapi', 'huggingface_hub')
# Set the environment variable to point the Hugging Face Hub cache to the persistent volume
image = image.env({'HF_HUB_CACHE': model_path})


with image.imports():
    # Import core libraries
    import torch
    import numpy as np

    # Import specialized libraries
    import sphn
    import sentencepiece

    # Import components for loading models
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders, LMGen


@app.cls(image=image, gpu='A10G', timeout=600, volumes={model_path: volume})
class Moshi:
    """
    A Modal class to run the Moshi model for real-time streaming audio processing,
    including audio-to-audio and audio-to-text generation.

    The application is deployed with a GPU ('A10G') and uses a persistent
    Modal Volume for caching model weights.
    """

    @modal.enter()
    def setup(self):
        """
        Initializes the model and sets up necessary components when the container starts.
        This includes:
        1. Determining the compute device (GPU/CPU).
        2. Downloading and loading the MIMI (audio encoder/decoder) model.
        3. Configuring MIMI's frame size and codebooks.
        4. Downloading and loading the MOSHI (language model/generator) model.
        5. Initializing the LMGen utility for sequence generation.
        6. Setting up streaming modes for both models.
        7. Downloading and loading the SentencePiece tokenizer for text output.
        8. Running a warm-up sequence to initialize model states and clear buffers.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Download and load the MIMI model weights
        mimi_weights = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weights, device=self.device)
        self.mimi.set_num_codebooks(8)
        # Calculate the size of an audio frame in samples
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        # Download and load the MOSHI Language Model weights
        moshi_weights = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        self.moshi = loaders.get_moshi_lm(moshi_weights, device=self.device)
        # Initialize the generator utility
        self.lm_gen = LMGen(self.moshi, temp=0.8, top_k=250)

        # Set models to streaming mode, indicating a batch size of 1
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Download and load the text tokenizer
        tokenizer_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # Warm-up loop: Run a few zero-audio chunks through the models
        # to pre-populate internal buffers and ensure a clean start for the first real request.
        for _ in range(4):
            # Create a zero-filled audio chunk
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk) # Encode the chunk to audio codes

            # Process the codes through the Language Model, codebook by codebook
            for c in range(codes.shape[-1]):
                output_tokens = self.lm_gen.step(codes[:, :, c:c+1])

                # If the LM generates an output sequence (contains new audio/text tokens)
                if output_tokens is not None:
                    # Decode the generated audio tokens (skipping the first token which is often text or a special token)
                    self.mimi.decode(output_tokens[:, 1:])

        # Synchronize CUDA to ensure the warm-up is complete before serving requests
        torch.cuda.synchronize()

    @modal.asgi_app()
    def web(self):
        """
        Defines the FastAPI application and the WebSocket endpoint for real-time
        audio processing.

        This sets up three concurrent tasks for each connection:
        1. `recieve()`: Reads incoming Opus audio bytes from the client.
        2. `process()`: Decodes the audio, runs the model, and generates output.
        3. `send()`: Sends generated audio (Opus bytes) and text back to the client.
        """
        from fastapi import FastAPI, WebSocket

        web_app = FastAPI()

        @web_app.websocket('/ws')
        async def websocket(ws: WebSocket):
            """
            The main WebSocket handler. Manages the connection and concurrent
            receive/process/send tasks.
            """
            await ws.accept()

            # Reset the streaming state for both models for a new connection
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            # Initialize Opus stream readers/writers for efficient audio handling
            opus_in = sphn.OpusStreamReader(self.mimi.sample_rate)
            opus_out = sphn.OpusStreamWriter(self.mimi.sample_rate)

            async def recieve():
                """Task to asynchronously receive Opus audio bytes from the client."""
                while True:
                    data = await ws.receive_bytes()

                    if isinstance(data, bytes) and len(data) > 0:
                        opus_in.append_bytes(data)

            async def process():
                """
                Task to process the received audio, run the Moshi model, and
                generate output audio and text.
                """
                pcm_buffer = None

                while True:
                    await asyncio.sleep(0.001) # Small sleep to yield control

                    # Read PCM (Pulse-Code Modulation) data from the Opus input stream
                    pcm = opus_in.read_pcm()

                    if pcm is None or len(pcm) == 0:
                        continue

                    # Buffer the received PCM data
                    pcm_buffer = pcm if pcm_buffer is None else np.concatenate([pcm_buffer, pcm])

                    # Process audio in fixed-size frames
                    while pcm_buffer.shape[-1] >= self.frame_size:
                        # Extract a frame and convert to a torch tensor on the correct device
                        chunk = torch.from_numpy(pcm_buffer[:self.frame_size]).to(self.device)[None, None]

                        # Update the buffer
                        pcm_buffer = pcm_buffer[self.frame_size:]

                        # Encode the audio chunk into discrete codes
                        codes = self.mimi.encode(chunk)

                        # Run the Language Model step-by-step for each codebook's codes
                        for c in range(codes.shape[-1]):
                            output_tokens = self.lm_gen.step(codes[:, :, c:c+1])

                            if output_tokens is not None:
                                # Decode the generated audio tokens back to PCM
                                # output_tokens[:, 1:] skips the first token, which is often text or a special token
                                audio = self.mimi.decode(output_tokens[:, 1:])
                                # Append the decoded audio to the Opus output stream
                                opus_out.append_pcm(audio[0,0].cpu().numpy())

                                # Check for generated text tokens (the first token in the sequence)
                                text_token = output_tokens[0,0,0].item()
                                # Text tokens are non-zero, non-EOS/BOS tokens
                                if text_token not in (0, 3):
                                    # Convert token ID back to text piece, handling SentencePiece's space convention
                                    text = self.text_tokenizer.id_to_piece(text_token).replace(' ', ' ')

                                    # Send text: '\x02' is a custom prefix for text data
                                    await ws.send_bytes(b'\x02' + text.encode('utf-8'))

            async def send():
                """Task to asynchronously read encoded Opus bytes and send them to the client."""
                while True:
                    await asyncio.sleep(0.001)

                    # Read encoded Opus bytes from the writer
                    audio = opus_out.read_bytes()

                    if audio and len(audio) > 0:
                        # Send audio: '\x01' is a custom prefix for audio data
                        await ws.send_bytes(b'\x01' + audio)

            try:
                # Disable gradient calculation during inference for performance and memory
                with torch.no_grad():
                    # Create and run all three tasks concurrently
                    tasks = [asyncio.create_task(recieve()), asyncio.create_task(process()), asyncio.create_task(send())]
                    await asyncio.gather(*tasks)
            except Exception as e:
                print(f'Error {e}, {str(e)}')
                # On error or connection close, cancel all running tasks
                for task in tasks:
                    task.cancel()

        return web_app