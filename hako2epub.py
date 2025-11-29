"""
hako2epub - A tool to download light novels from ln.hako.vn / docln.net
"""

import argparse
import logging
import sys
from os.path import abspath, dirname

# Add the library path to sys.path
sys.path.insert(0, abspath(dirname(__file__)))

from hako2epub_lib.app import Application

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Hako2Epub Downloader")
    parser.add_argument("url", nargs="?", help="The URL of the light novel to download")
    args = parser.parse_args()

    try:
        app = Application(args.url)
        app.run()
    except KeyboardInterrupt:
        print("\nExit.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()