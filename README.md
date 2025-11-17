# Moshi Real-Time Voice Chat on Modal

A streaming, full-duplex voice-to-voice and voice-to-text application powered by the **Moshi** multimodal model, deployed in real-time on **Modal**.

This project uses **WebSockets** for low-latency communication, processes audio on an **NVIDIA A10G GPU**, and provides a simple React frontend for immediate interaction.

---

## 🚀 Key Features

* **Real-Time Streaming:** Uses WebSockets for low-latency, bidirectional audio/text.
* **GPU Acceleration:** Moshi model runs on Modal's **A10G GPU**.
* **Caching:** Uses a **Modal Volume** to cache model weights, speeding up subsequent starts.
* **Full Pipeline:** Implements the complete audio-to-audio/text Moshi processing chain.
* **Frontend:** Simple React client handles Opus encoding/decoding and gapless playback.

---

## 📁 Project Structure

| File | Role |
| :--- | :--- |
| `moshi.py` | **Model Service:** Modal Class for GPU model loading and core WebSocket processing logic. |
| `app.py` | **Web Server:** Modal function that serves the static React frontend via FastAPI. |
| `common.py` | Initializes the shared `modal.App` object. |
| `frontend/` | Static files (HTML, React/JSX) for the user interface. |

---

## 🛠️ Deployment

1.  **Prerequisite:** Install and configure the Modal CLI (`pip install modal`, `python -m modal setup`).
2.  **Deploy:** Run the deployment command from the project root:
    ```bash
    modal -m serve folder.app
    ```
3.  **Access:** Navigate to the public URL provided by Modal. Grant microphone access and start speaking for real-time output.
