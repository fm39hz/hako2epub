"""
hako2epub - A tool to download light novels from ln.hako.vn / docln.net
Supports downloading individual volumes or merging them into a single EPUB.
"""

import argparse
import json
import re
import time
import logging
from io import BytesIO
from multiprocessing.dummy import Pool as ThreadPool
from os import mkdir
from os.path import isdir, isfile, join
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import questionary
import requests
import tqdm
from bs4 import BeautifulSoup
from ebooklib import epub
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DOMAINS = ["ln.hako.vn", "docln.net", "docln.sbs"]
SLEEP_TIME = 30
LINE_SIZE = 80
THREAD_NUM = 8
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.97 Safari/537.36",
    "Referer": "https://docln.net/",
}
TOOL_VERSION = "2.2.0"
HTML_PARSER = "html.parser"

# Session for requests
session = requests.Session()


@dataclass
class Chapter:
    """Represents a chapter in a light novel."""

    name: str
    url: str


@dataclass
class Volume:
    """Represents a volume in a light novel."""

    url: str = ""
    name: str = ""
    cover_img: str = ""
    num_chapters: int = 0
    # CHANGED: Use List instead of Dict to prevent overwriting duplicate chapter names (Logic from Colab)
    chapters: List[Chapter] = field(default_factory=list)


@dataclass
class LightNovel:
    """Represents a light novel with all its information."""

    name: str = ""
    url: str = ""
    author: str = "Unknown"
    summary: str = ""
    series_info: str = ""
    main_cover: str = ""  # Added from Colab logic
    volumes: List[Volume] = field(default_factory=list)


class ColorCodes:
    """ANSI color codes for terminal output."""

    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    FAIL = "\03[91m"
    ENDC = "\033[0m"


class NetworkManager:
    """Handles network requests with retry logic (Local Script Logic)."""

    @staticmethod
    def check_available_request(url: str, stream: bool = False) -> requests.Response:
        if not url.startswith("http"):
            url = "https://" + url if not url.startswith("//") else "https:" + url

        # Simple domain rotation logic
        original_url = url
        domains_to_try = DOMAINS[:]

        # Determine path
        path = url
        for domain in DOMAINS:
            if domain in url:
                # simple split to find path
                parts = url.split(domain)
                if len(parts) > 1:
                    path = parts[1]
                break

        last_exception = None

        for domain in domains_to_try:
            # Reconstruct URL with current domain trial
            if any(d in original_url for d in DOMAINS):
                # Replace existing domain
                url = f"https://{domain}{path}"

            headers = HEADERS.copy()
            headers["Referer"] = f"https://{domain}"

            for retry in range(3):
                try:
                    response = session.get(
                        url, stream=stream, headers=headers, timeout=30
                    )
                    if response.status_code == 200:
                        return response
                    elif response.status_code == 404:
                        break  # Don't retry 404
                    else:
                        time.sleep(2)
                except requests.RequestException as e:
                    last_exception = e
                    time.sleep(2)

            # If 404 or success, break loop, else try next domain
            if "response" in locals() and response.status_code == 200:
                return response

        if last_exception:
            raise last_exception
        raise requests.RequestException(f"Failed to access {original_url}")


class TextUtils:
    @staticmethod
    def format_text(text: str) -> str:
        return text.strip().replace("\n", "")

    @staticmethod
    def format_filename(name: str) -> str:
        # Strict sanitization
        name = re.sub(r'[\\/*?:"<>|]', "", name)
        name = name.replace(" ", "_")
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

        if url.startswith("/"):
            return f"https://{domain}{url}"
        return f"https://{domain}/{url}"


class ImageManager:
    @staticmethod
    def get_image(image_url: str) -> Optional[Image.Image]:
        if not image_url:
            return None
        if "imgur.com" in image_url and "." not in image_url[-5:]:
            image_url += ".jpg"

        try:
            response = NetworkManager.check_available_request(image_url, stream=True)
            image = Image.open(response.raw).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Cannot get image: {image_url}")
            return None


class OutputFormatter:
    @staticmethod
    def print_formatted(name: str, info: str, info_style="bold fg:cyan"):
        questionary.print("! ", style="bold fg:gray", end="")
        questionary.print(name, style="bold fg:white", end="")
        questionary.print(info, style=info_style)

    @staticmethod
    def print_success(message: str, item_name: str = ""):
        print(
            f"{message} {ColorCodes.OKCYAN}{item_name}{ColorCodes.ENDC}: [{ColorCodes.OKGREEN} DONE {ColorCodes.ENDC}]"
        )


class EpubEngine:
    def __init__(self):
        self.book = None
        self.light_novel = None
        self.volume = None
        # CSS from Colab
        self.css_style = """
            body { margin: 0; padding: 5px; text-align: justify; line-height: 1.4em; }
            h1, h2, h3 { text-align: center; margin: 1em 0; font-weight: bold; }
            p { margin-top: 0; margin-bottom: 0.5em; text-indent: 0; }
            img { display: block; margin: 10px auto; max-width: 100%; height: auto; page-break-inside: avoid; }
            .center { text-align: center; }
            .summary { margin: 20px; font-style: italic; background-color: #f9f9f9; padding: 10px; border-radius: 5px; }
        """

    def make_cover_image(self, url: str) -> Optional[epub.EpubItem]:
        if not url:
            return None
        try:
            image = ImageManager.get_image(url)
            if not image:
                return None

            b = BytesIO()
            image.save(b, "jpeg")
            return epub.EpubItem(
                file_name="cover_image.jpg",
                media_type="image/jpeg",
                content=b.getvalue(),
            )
        except:
            return None

    def make_intro_page(self, is_merged=False) -> epub.EpubHtml:
        """Adapted from Colab's intro HTML structure."""
        intro_html = f"""
            <div style="text-align: center; margin-top: 5%;">
                <h1>{self.light_novel.name}</h1>
        """

        if not is_merged and self.volume:
            intro_html += f"<h3>{self.volume.name}</h3>"

        intro_html += f"""
                <h3>Tác giả: {self.light_novel.author}</h3>
                <hr style="width: 50%;"/>
                {self.light_novel.series_info}
                <div style="text-align: justify; margin: 30px 10px;">
                    <h4 style="text-align: center">Tóm tắt</h4>
                    {self.light_novel.summary}
                </div>
            </div>
        """

        # Cover logic
        cover_url = (
            self.light_novel.main_cover
            if is_merged
            else (self.volume.cover_img if self.volume else "")
        )
        cover_item = self.make_cover_image(cover_url)

        if cover_item:
            self.book.add_item(cover_item)
            self.book.set_cover("cover.jpg", cover_item.content)
            # Add image tag to intro html
            img_tag = f'<img src="cover_image.jpg" alt="Cover" style="max-height: 600px; object-fit: contain;">'
            intro_html = img_tag + intro_html

        return epub.EpubHtml(
            title="Giới thiệu",
            file_name="intro.xhtml",
            content=intro_html,
            uid="intro_page",
        )

    def _process_chapter_content(
        self, data: Tuple
    ) -> Optional[Tuple[int, epub.EpubHtml]]:
        """Process a single chapter. Adapted from Colab logic but using Local Network."""
        idx, name, url, file_prefix = data
        try:
            resp = NetworkManager.check_available_request(url)
            soup = BeautifulSoup(resp.text, HTML_PARSER)

            content_div = soup.find("div", id="chapter-content")
            if not content_div:
                return None

            # Clean content (Colab logic)
            for bad in content_div.find_all(
                ["div", "p", "a"],
                class_=["d-none", "d-md-block", "flex", "note-content"],
            ):
                bad.decompose()

            # Process Images
            for i, img in enumerate(content_div.find_all("img")):
                img_src = img.get("src")
                if not img_src or "chapter-banners" in img_src:
                    img.decompose()
                    continue

                img_obj = ImageManager.get_image(img_src)
                if img_obj:
                    img_filename = f"{file_prefix}img_{idx}_{i}.jpg"
                    b = BytesIO()
                    img_obj.save(b, "jpeg")

                    epub_img = epub.EpubItem(
                        file_name=f"images/{img_filename}",
                        media_type="image/jpeg",
                        content=b.getvalue(),
                    )
                    self.book.add_item(epub_img)

                    img["src"] = f"images/{img_filename}"
                    if "style" in img.attrs:
                        del img["style"]

            # Clean empty tags
            for element in content_div.find_all(["p", "div", "span", "br"]):
                if not element.get_text(strip=True) and not element.find("img"):
                    element.decompose()

            # Handle Notes (Local logic)
            note_divs = soup.find_all("div", id=re.compile("^note"))
            content_str = str(content_div)
            for div in note_divs:
                nid = div.get("id")
                ncontent = div.find("span", class_="note-content_real")
                if nid and ncontent:
                    content_str = content_str.replace(
                        f"[{nid}]", f"(Note: {ncontent.text})"
                    )

            # Final HTML
            html_content = f"""
                <h2>{name}</h2>
                <div id="content">{content_str}</div>
            """

            fname = f"{file_prefix}chap_{idx}.xhtml"
            chapter_item = epub.EpubHtml(
                title=name, file_name=fname, content=html_content, uid=fname
            )
            chapter_item.add_link(
                href="style/nav.css", rel="stylesheet", type="text/css"
            )

            return (idx, chapter_item)

        except Exception as e:
            logger.error(f"Error processing chapter {url}: {e}")
            return None

    def make_chapters(
        self, volume: Volume, global_start_index: int = 1, file_prefix: str = ""
    ) -> List[epub.EpubHtml]:
        # Convert List[Chapter] to data tuples for processing
        tasks = []
        for i, chapter in enumerate(volume.chapters):
            tasks.append(
                (global_start_index + i, chapter.name, chapter.url, file_prefix)
            )

        pool = ThreadPool(THREAD_NUM)
        results = []
        try:
            print(f"Downloading {len(tasks)} chapters from {volume.name}...")
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(self._process_chapter_content, tasks),
                    total=len(tasks),
                )
            )
        finally:
            pool.close()
            pool.join()

        # Sort by index
        results = sorted([r for r in results if r], key=lambda x: x[0])
        return [r[1] for r in results]

    def create_epub(self, ln: LightNovel):
        """Standard mode: One EPUB per volume."""
        self.light_novel = ln

        css_item = epub.EpubItem(
            uid="style_nav",
            file_name="style/nav.css",
            media_type="text/css",
            content=self.css_style,
        )

        for volume in ln.volumes:
            fname = TextUtils.format_filename(f"{volume.name} - {ln.name}") + ".epub"
            fpath = join("downloaded", TextUtils.format_filename(ln.name), fname)

            if isfile(fpath):
                print(f"Skipping {fname} (Already exists)")
                continue

            self.book = epub.EpubBook()
            self.book.set_identifier(volume.url)
            self.book.set_title(f"{volume.name} - {ln.name}")
            self.book.set_language("vi")
            self.book.add_author(ln.author)
            self.book.add_item(css_item)
            self.volume = volume

            intro = self.make_intro_page(is_merged=False)
            intro.add_item(css_item)
            self.book.add_item(intro)

            chapters = self.make_chapters(volume)
            for c in chapters:
                self.book.add_item(c)

            self.book.spine = ["cover", intro, "nav"] + chapters
            self.book.add_item(epub.EpubNcx())
            self.book.add_item(epub.EpubNav())

            self._write_file(fpath)

    def create_merged_epub(self, ln: LightNovel, selected_volumes: List[Volume]):
        """Merge mode: One EPUB for all selected volumes."""
        self.light_novel = ln
        self.book = epub.EpubBook()
        self.book.set_identifier(ln.url)
        self.book.set_title(f"{ln.name} [Merged]")
        self.book.set_language("vi")
        self.book.add_author(ln.author)

        css_item = epub.EpubItem(
            uid="style_nav",
            file_name="style/nav.css",
            media_type="text/css",
            content=self.css_style,
        )
        self.book.add_item(css_item)

        # Global Intro
        intro = self.make_intro_page(is_merged=True)
        intro.add_item(css_item)
        self.book.add_item(intro)

        spine = ["cover", intro, "nav"]
        toc = [intro]

        global_idx = 1

        for i, volume in enumerate(selected_volumes):
            self.volume = volume
            OutputFormatter.print_formatted("Processing Volume", volume.name)

            # Volume Separator Page
            vol_id = f"vol_{i + 1}"
            vol_html = f"""
                <div style="text-align: center; margin-top: 30vh;">
                    <h1>{volume.name}</h1>
                </div>
            """
            vol_page = epub.EpubHtml(
                title=volume.name,
                file_name=f"{vol_id}.xhtml",
                content=vol_html,
                uid=vol_id,
            )
            vol_page.add_item(css_item)
            self.book.add_item(vol_page)
            spine.append(vol_page)

            # Chapters
            # We add a prefix (v1_, v2_) to filenames to ensure uniqueness in the merged file
            vol_prefix = f"v{i + 1}_"
            chapters = self.make_chapters(
                volume, global_start_index=global_idx, file_prefix=vol_prefix
            )

            for c in chapters:
                self.book.add_item(c)
            spine.extend(chapters)

            # Nested TOC entry: (Volume Page, [List of Chapters])
            toc.append((vol_page, chapters))

            global_idx += len(chapters) + 1
            print("-" * LINE_SIZE)

        self.book.spine = spine
        self.book.toc = toc
        self.book.add_item(epub.EpubNcx())
        self.book.add_item(epub.EpubNav())

        fname = TextUtils.format_filename(f"{ln.name} - Merged") + ".epub"
        fpath = join("downloaded", TextUtils.format_filename(ln.name), fname)
        self._write_file(fpath)

    def _write_file(self, path):
        folder = join("downloaded", TextUtils.format_filename(self.light_novel.name))
        if not isdir("downloaded"):
            mkdir("downloaded")
        if not isdir(folder):
            mkdir(folder)

        try:
            epub.write_epub(path, self.book, {})
            OutputFormatter.print_success("Saved EPUB", path)
        except Exception as e:
            print(f"Error saving file: {e}")


class LightNovelManager:
    def parse_ln_info(self, url: str) -> Optional[LightNovel]:
        """
        Parses Light Novel info using the logic from the Colab notebook.
        Includes extracting main cover from style attributes and using Lists for chapters.
        """
        try:
            print("Fetching Novel Info...", end="\r")
            resp = NetworkManager.check_available_request(url)
            soup = BeautifulSoup(resp.text, HTML_PARSER)

            ln = LightNovel()
            ln.url = url

            # Name
            name_tag = soup.find("span", "series-name")
            ln.name = name_tag.text.strip() if name_tag else "Unknown"

            # Main Cover (Regex approach from Colab)
            main_cover_div = soup.find("div", "series-cover")
            if main_cover_div:
                img_div = main_cover_div.find("div", "img-in-ratio")
                if img_div and "style" in img_div.attrs:
                    style = img_div["style"]
                    match = re.search(r'url\([\'"]?([^\'"\)]+)[\'"]?\)', style)
                    if match:
                        ln.main_cover = match.group(1)

            # Series Info & Author
            series_info = soup.find("div", "series-information")
            if series_info:
                # Remove links for clean info
                for a in series_info.find_all("a"):
                    if "href" in a.attrs:
                        del a.attrs["href"]
                ln.series_info = str(series_info)

                for item in series_info.find_all("div", "info-item"):
                    label = item.find(class_="info-name")
                    if label and "Tác giả" in label.text:
                        val = item.find(class_="info-value")
                        if val:
                            ln.author = val.text.strip()

            # Summary (Cleaned)
            summary_div = soup.find("div", "summary-content")
            if summary_div:
                for bad in summary_div.find_all(
                    ["a", "div", "span"],
                    class_=["see-more", "less-state", "more-state"],
                ):
                    bad.decompose()
                ln.summary = "".join([str(x) for x in summary_div.contents]).strip()

            # Volumes (List Structure)
            vol_sections = soup.find_all("section", "volume-list")
            for sect in vol_sections:
                vol = Volume()
                title_elem = sect.find("span", "sect-title")
                vol.name = title_elem.text.strip() if title_elem else "Unknown Volume"

                # Volume Cover
                cover_div = sect.find("div", "volume-cover")
                if cover_div:
                    img_div = cover_div.find("div", "img-in-ratio")
                    if img_div and "style" in img_div.attrs:
                        match = re.search(
                            r'url\([\'"]?([^\'"\)]+)[\'"]?\)', img_div["style"]
                        )
                        if match:
                            vol.cover_img = match.group(1)

                    a_tag = cover_div.find("a")
                    if a_tag:
                        vol.url = TextUtils.reformat_url(url, a_tag["href"])

                # Chapters (Append to List)
                chap_list = sect.find("ul", "list-chapters")
                if chap_list:
                    for li in chap_list.find_all("li"):
                        a = li.find("a")
                        if a:
                            c_url = TextUtils.reformat_url(url, a["href"])
                            # Colab logic: Append object/dict to list
                            vol.chapters.append(Chapter(name=a.text.strip(), url=c_url))

                ln.volumes.append(vol)

            OutputFormatter.print_success("Fetched Info", ln.name)
            return ln

        except Exception as e:
            logger.error(f"Parse Error: {e}")
            return None

    def start(self, ln_url: str, is_merge: bool):
        ln = self.parse_ln_info(ln_url)
        if not ln or not ln.volumes:
            print("Failed to parse novel or no volumes found.")
            return

        print(f"Novel: {ln.name} | Volumes: {len(ln.volumes)}")

        # Selection UI
        choices = [f"{v.name} ({len(v.chapters)} chapters)" for v in ln.volumes]

        if is_merge:
            selected_indices = questionary.checkbox(
                "Select volumes to MERGE into one file:", choices=choices
            ).ask()
        else:
            choices.insert(0, "All volumes")
            selected_indices = questionary.checkbox(
                "Select volumes to Download (Separate files):", choices=choices
            ).ask()

        if not selected_indices:
            print("No volumes selected.")
            return

        engine = EpubEngine()

        if is_merge:
            # Filter volumes based on selection text
            selected_vols = []
            for i, c in enumerate(choices):
                if c in selected_indices:
                    selected_vols.append(ln.volumes[i])

            engine.create_merged_epub(ln, selected_vols)
        else:
            # Standard Mode
            if "All volumes" in selected_indices:
                engine.create_epub(ln)
            else:
                # Filter volumes
                target_vols = []
                # choices has 'All volumes' at index 0, so offset is needed
                real_choices = choices[1:]
                for i, c in enumerate(real_choices):
                    if c in selected_indices:
                        target_vols.append(ln.volumes[i])

                ln.volumes = target_vols
                engine.create_epub(ln)


def main():
    parser = argparse.ArgumentParser(description="Hako2Epub Downloader")
    parser.add_argument("url", nargs="?", help="URL of the light novel")
    parser.add_argument(
        "-m", "--merge", action="store_true", help="Merge mode (Single File)"
    )

    args = parser.parse_args()

    manager = LightNovelManager()

    target_url = args.url
    is_merge = args.merge

    if not target_url:
        target_url = questionary.text("Enter Light Novel URL:").ask()
        if not is_merge:
            is_merge = questionary.confirm(
                "Do you want to MERGE volumes into one file?"
            ).ask()

    if target_url:
        manager.start(target_url, is_merge)


if __name__ == "__main__":
    main()
