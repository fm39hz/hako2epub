# hako2epub

A Python-based tool to download light novels from Hako (docln.net, ln.hako.vn) and convert them into EPUB format for offline reading.

## Features

- **Interactive Downloader**: Guides you through selecting a novel and choosing which volumes to download.
- **EPUB Generation**: Converts the downloaded content into well-formatted EPUB files.
- **Flexible Build Options**:
  - **Merged EPUB**: Combine all selected volumes into a single, comprehensive EPUB file.
  - **Separate EPUBs**: Create an individual EPUB file for each volume.
- **Smart Caching**: Saves downloaded chapter data locally. It intelligently re-downloads only if the cached data is missing or incomplete, saving time and bandwidth.
- **Image and Footnote Handling**: Downloads and embeds images within the chapters and correctly processes footnotes for a clean reading experience.
- **Resilient Networking**: Automatically retries downloads and cycles through available domains to handle network errors and site changes gracefully.

## Usage

1. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the script:**

    ```bash
    python hako2epub.py
    ```

    Or provide a URL directly:

    ```bash
    python hako2epub.py "https://docln.net/truyen/some-novel"
    ```

3. **Follow the on-screen prompts** to select an action (Download, Build, or Full Process) and choose the desired volumes.

The script will create a `saved_data` directory to store the downloaded content and a `downloaded` directory for the final EPUB files.
