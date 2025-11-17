from pathlib import Path

import modal

# Import the shared Modal App object and the Moshi class definition
from .common import app
from .moshi import Moshi


# Define the path to the 'frontend' directory relative to the current file (app.py)
frontend = Path(__file__).with_name('frontend')
# Build the base image for the FastAPI web server
image = modal.Image.debian_slim(python_version='3.11').pip_install('fastapi').add_local_dir(frontend, '/assets')
"""
Modal Image for the frontend web application.

It is based on debian-slim, installs FastAPI, and copies the local 'frontend' 
directory content into the container's '/assets' directory to serve static files.
"""


@app.function(image=image, timeout=600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    """
    Defines the web application that serves the static frontend files.

    This function is configured as:
    1. A Modal function using the `image` defined above.
    2. Concurrent, allowing up to 100 simultaneous inputs (connections).
    3. An ASGI application, making it a web endpoint.

    It sets up a FastAPI application, adds CORS middleware for broad compatibility,
    and mounts the static files from the '/assets' directory to the root path ('/').
    This ensures that when a user navigates to the deployment URL, the frontend
    HTML/JS/CSS is served.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles

    webapp = FastAPI()

    # Configure and add CORS middleware to allow requests from any origin.
    # This is often necessary for web sockets and development environments.
    webapp.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
        allow_credentials=True
    )

    # Mount the StaticFiles handler to serve the frontend assets.
    # The content of the 'frontend' directory (copied to '/assets' in the image)
    # is served from the root URL ('/'). The 'html=True' flag ensures
    # 'index.html' is served for the root path.
    webapp.mount('/', StaticFiles(directory='/assets', html=True))

    return webapp