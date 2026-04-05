# server/app.py — OpenEnv multi-mode deployment entry point
# Required by openenv validate multi-mode check.
# The actual application logic lives in the root server.py.

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401 — re-export the FastAPI app


def main():
    """Entry point for the server script (required by openenv validate)."""
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()
