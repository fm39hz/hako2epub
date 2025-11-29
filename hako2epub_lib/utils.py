import json
import logging
import re
from os.path import exists
from typing import List

from .constants import BOOKS_FILE, DOMAINS

logger = logging.getLogger(__name__)


# --- Book List Management ---


def read_books_list() -> List[str]:
    """Reads the list of book folders from books.json."""
    if not exists(BOOKS_FILE):
        with open(BOOKS_FILE, "w", encoding="utf-8") as f:
            json.dump({"books": []}, f)
        return []
    try:
        with open(BOOKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("books", [])
    except (json.JSONDecodeError, IOError):
        return []


def write_books_list(books: List[str]):
    """Writes the list of book folders to books.json."""
    with open(BOOKS_FILE, "w", encoding="utf-8") as f:
        json.dump({"books": sorted(list(set(books)))}, f, ensure_ascii=False, indent=2)

def add_book_to_list(book_folder: str):
    """Adds a book folder to the books.json list if it's not already there."""
    books = read_books_list()
    if book_folder not in books:
        books.append(book_folder)
        write_books_list(books)
        logger.info(f"Added '{book_folder}' to {BOOKS_FILE}")


class TextUtils:
    @staticmethod
    def format_filename(name: str) -> str:
        name = re.sub(r'[\\/*?:\"<>|]', "", name)
        name = name.replace(" ", "_").strip()
        return name[:100]

    @staticmethod
    def reformat_url(base_url: str, url: str) -> str:
        if url.startswith("http"):
            return url
        domain = "docln.net"
        for d in DOMAINS:
            if d in base_url:
                domain = d
                break
        return (
            f"https://{domain}{url}"
            if url.startswith("/")
            else f"https://{domain}/{url}"
        )
