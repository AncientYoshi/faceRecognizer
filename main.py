"""Convenience entry point for running the service from an IDE."""

import uvicorn

from app.main import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
