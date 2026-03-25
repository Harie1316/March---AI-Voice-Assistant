"""
Application entry point for the AI Voice Assistant.
Run with: python run.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from config.settings import settings


def main():
    app = create_app()
    app.run(
        host=settings.flask.host,
        port=settings.flask.port,
        debug=settings.flask.debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
