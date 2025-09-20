# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hako2epub is a Python tool for downloading light novels from Vietnamese light novel websites (ln.hako.vn, docln.net, docln.sbs) and converting them to EPUB format. The tool supports downloading complete novels, specific volumes, or individual chapters, with features like image processing, table of contents generation, and update management.

## Commands

### Running the Tool
```bash
# Download a complete light novel
python hako2epub.py <light_novel_url>

# Download specific chapters
python hako2epub.py -c <light_novel_url>

# Update all downloaded light novels
python hako2epub.py -u

# Update a specific light novel
python hako2epub.py -u <light_novel_url>

# Show version
python hako2epub.py -v
```

### Dependencies Installation
```bash
pip install ebooklib requests bs4 pillow argparse tqdm questionary
```

### Testing the Tool
```bash
# Test with a sample light novel URL (manual testing required)
python hako2epub.py https://ln.hako.vn/truyen/example-novel

# Check tool version and help
python hako2epub.py --help
python hako2epub.py --version
```

## Architecture

### Core Components

1. **LightNovelManager** (`hako2epub.py:1152-1586`): Main orchestrator that handles URL validation, domain checking, and coordinates the download process
2. **EpubEngine** (`hako2epub.py:756-1149`): Handles EPUB file creation, chapter processing, image embedding, and metadata management
3. **UpdateManager** (`hako2epub.py:317-753`): Manages updating existing light novels with new chapters/volumes
4. **NetworkManager** (`hako2epub.py:98-186`): Handles HTTP requests with retry logic and domain failover
5. **ImageManager** (`hako2epub.py:253-277`): Processes and downloads images for embedding in EPUBs

### Data Models
- **LightNovel** (`hako2epub.py:72-82`): Contains novel metadata (name, author, summary, volumes)
- **Volume** (`hako2epub.py:62-69`): Represents a volume with chapters and cover image
- **Chapter** (`hako2epub.py:55-59`): Simple name-URL pair for individual chapters

### Key Features
- **Multi-domain support**: Automatically switches between ln.hako.vn, docln.net, and docln.sbs
- **Rate limiting**: Pauses for 120 seconds after 190 requests to avoid blocking
- **Concurrent processing**: Uses ThreadPool with 8 threads for chapter downloads
- **Image processing**: Downloads and embeds images, handles various image hosts
- **Persistence**: Tracks downloads in `downloaded/ln_info.json` for update functionality
- **Interactive CLI**: Uses questionary for volume/chapter selection

### File Structure
- Single-file application: `hako2epub.py` (1618 lines)
- Configuration stored in: `downloaded/ln_info.json` (created automatically)
- Output: Creates 'downloaded' directory with subfolders per novel containing EPUB files
- Images embedded directly in EPUB files

### Error Handling
- Network errors trigger domain failover and retries
- Image download failures are logged but don't stop processing
- Malformed HTML is handled gracefully with fallback values
- JSON parsing errors are caught and reported

## Development Notes

- No formal test suite - testing requires manual verification with actual URLs
- No linting configuration present - code follows basic Python conventions
- No CI/CD pipeline - releases are manual
- Heavy use of BeautifulSoup for HTML parsing
- Extensive logging for debugging network and parsing issues