"""
Entry point: python -m nadiru_engine
"""

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from .service import app

if __name__ == "__main__":
    host = os.getenv("ENGINE_HOST", "0.0.0.0")
    port = int(os.getenv("ENGINE_PORT", "8765"))
    print(f"Starting Nadiru engine on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
