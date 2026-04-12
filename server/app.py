"""
Server entry point for OpenEnv compatibility.
Re-exports the FastAPI app from the firmware_debug_env package.
"""

import uvicorn
from firmware_debug_env.server.app import app  # noqa: F401


def main():
    """Run the firmware debug environment server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
