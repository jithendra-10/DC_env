# server/app.py — OpenEnv multi-mode deployment entry point
# This file is required by the openenv validate multi-mode check.
# The actual application logic lives in the root server.py.

import sys
import os

# Add the project root to the path so we can import from it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401 — re-export the FastAPI app

__all__ = ["app"]
