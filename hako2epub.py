"""
hako2epub - A tool to download light novels from ln.hako.vn / docln.net
"""

import argparse
import json
import re
import time
import logging
import os
import shutil
import io
import html
from multiprocessing.dummy import Pool as ThreadPool
from os.path import isdir, join, exists, splitext
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse

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
DOMAINS = ["docln.net", "ln.hako.vn", "docln.sbs"]
IMAGE_DOMAINS = [
    "i.hako.vip",
    "i.docln.net",
    "ln.hako.vn",
    "i2.docln.net",
    "i2.hako.vip",
]
DATA_DIR = "data"
RESULT_DIR = "result"

THREAD_NUM = 1
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://docln.net/",
}
HTML_PARSER = "html.parser"

session = requests.Session()

# --- Data Structures ---


@dataclass
class Chapter:
    name: str
    url: str


@dataclass
class Volume:
    url: str = ""
    name: str = ""
    cover_img: str = ""
    chapters: List[Chapter] = field(default_factory=list)


@dataclass
class LightNovel:
    name: str = ""
    url: str = ""
    author: str = "Unknown"
    summary: str = ""
    main_cover: str = ""
    tags: List[str] = field(default_factory=list)
    volumes: List[Volume] = field(default_factory=list)


# --- Utilities ---


class TextUtils:
    @staticmethod
    def format_filename(name: str) -> str:
        name = re.sub(r'[\\/*?:"<>|]', "", name)
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


class NetworkManager:
    REQUEST_COUNT = 0

    @staticmethod
    def is_internal_domain(url: str) -> bool:
        parsed = urlparse(url)
        domain = parsed.netloc
        all_internal = DOMAINS + IMAGE_DOMAINS
        return any(d in domain for d in all_internal)

    @staticmethod
    def check_available_request(url: str, stream: bool = False) -> requests.Response:
        NetworkManager.REQUEST_COUNT += 1
        if NetworkManager.REQUEST_COUNT > 0 and NetworkManager.REQUEST_COUNT % 100 == 0:
            logger.info("Anti-Ban: Pausing for 60 seconds...")
            time.sleep(60)

        if "i2.docln.net" in url:
            url = url.replace("i2.docln.net", "i2.hako.vip")

        if not url.startswith("http"):
            url = "https://" + url if not url.startswith("//") else "https:" + url

        parsed = urlparse(url)
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query

        is_image = (
            "/covers/" in path
            or path.endswith((".jpg", ".png", ".gif", ".jpeg"))
            or "/img/" in path
        )

        if not NetworkManager.is_internal_domain(url) and not is_image:
            headers = HEADERS.copy()
            if "Referer" in headers:
                del headers["Referer"]

            for i in range(3):
                try:
                    response = session.get(
                        url, stream=stream, headers=headers, timeout=30
                    )
                    if response.status_code == 200:
                        return response
                    elif response.status_code == 404:
                        break
                    time.sleep(1)
                except Exception:
                    time.sleep(1)
            raise requests.RequestException(f"Failed external link: {url}")

        domains_to_try = IMAGE_DOMAINS[:] if is_image else DOMAINS[:]
        original = parsed.netloc

        if original not in domains_to_try:
            domains_to_try.insert(0, original)
        else:
            domains_to_try.remove(original)
            domains_to_try.insert(0, original)

        last_exception = None

        for domain in domains_to_try:
            target_url = f"https://{domain}{path}"
            headers = HEADERS.copy()
            headers["Referer"] = f"https://{DOMAINS[0]}/"

            for i in range(3):
                try:
                    response = session.get(
                        target_url, stream=stream, headers=headers, timeout=30
                    )
                    if response.status_code == 200:
                        return response
                    elif response.status_code in [404, 403]:
                        break
                    time.sleep(2 + i)
                except requests.RequestException as e:
                    last_exception = e
                    time.sleep(2 + i)

            if "response" in locals() and response.status_code == 200:
                return response

        if last_exception:
            raise last_exception
        raise requests.RequestException(f"Failed to access {url}")

    @staticmethod
    def download_image_to_disk(url: str, save_path: str) -> bool:
        if not url:
            return False

        if exists(save_path):
            if os.path.getsize(save_path) > 0:
                return True
            else:
                os.remove(save_path)

        if "imgur.com" in url and "." not in url[-5:]:
            url += ".jpg"

        try:
            resp = NetworkManager.check_available_request(url, stream=True)
            with open(save_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
            return True
        except Exception as e:
            logger.warning(f"Image DL fail: {url} | {e}")
            return False


# --- Phase 1: Downloader ---


class NovelDownloader:
    def __init__(self, ln: LightNovel, base_folder: str):
        self.ln = ln
        self.base_folder = base_folder
        self.images_folder = join(base_folder, "images")
        if not exists(self.base_folder):
            os.makedirs(self.base_folder)
        if not exists(self.images_folder):
            os.makedirs(self.images_folder)

    def create_metadata_file(self):
        print(f"Updating metadata for: {self.ln.name}")
        local_cover_path = ""
        if self.ln.main_cover:
            ext = "jpg"
            if "png" in self.ln.main_cover:
                ext = "png"
            elif "gif" in self.ln.main_cover:
                ext = "gif"

            fname = f"main_cover.{ext}"
            save_path = join(self.images_folder, fname)
            if NetworkManager.download_image_to_disk(self.ln.main_cover, save_path):
                local_cover_path = f"images/{fname}"

        volume_list = []
        for i, vol in enumerate(self.ln.volumes):
            volume_list.append(
                {
                    "order": i + 1,
                    "name": vol.name,
                    "filename": TextUtils.format_filename(vol.name) + ".json",
                    "url": vol.url,
                }
            )

        metadata = {
            "novel_name": self.ln.name,
            "author": self.ln.author,
            "tags": self.ln.tags,
            "summary": self.ln.summary,
            "cover_image_local": local_cover_path,
            "url": self.ln.url,
            "volumes": volume_list,
        }

        with open(join(self.base_folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _process_chapter(self, data: Tuple[int, Chapter, str]) -> Dict:
        idx, chapter, img_prefix = data

        try:
            resp = NetworkManager.check_available_request(chapter.url)
            time.sleep(0.5)
            soup = BeautifulSoup(resp.text, HTML_PARSER)
            content_div = soup.find("div", id="chapter-content")
            if not content_div:
                return None

            for bad in content_div.find_all(
                ["div", "p", "a"],
                class_=["d-none", "d-md-block", "flex", "note-content"],
            ):
                bad.decompose()

            # Process Images
            for i, img in enumerate(content_div.find_all("img")):
                src = img.get("src")
                if not src or "chapter-banners" in src:
                    img.decompose()
                    continue

                ext = "jpg"
                if "png" in src:
                    ext = "png"
                elif "gif" in src:
                    ext = "gif"
                elif "webp" in src:
                    ext = "webp"

                local_name = f"{img_prefix}_chap_{idx}_img_{i}.{ext}"

                if NetworkManager.download_image_to_disk(
                    src, join(self.images_folder, local_name)
                ):
                    img["src"] = f"images/{local_name}"
                    if "style" in img.attrs:
                        del img["style"]
                    if "onclick" in img.attrs:
                        del img["onclick"]
                else:
                    img.decompose()

            for el in content_div.find_all(["p", "div", "span"]):
                if not el.get_text(strip=True) and not el.find("img"):
                    el.decompose()

            # Footnotes
            note_map = {}
            note_divs = list(soup.find_all("div", id=re.compile(r"^note\d+")))

            for div in note_divs:
                nid = div.get("id")
                content_span = div.find("span", class_="note-content_real")
                if content_span:
                    note_map[nid] = content_span.get_text().strip()
                div.decompose()

            note_reg = soup.find("div", class_="note-reg")
            if note_reg:
                note_reg.decompose()

            html_content = str(content_div)

            footnote_counter = 1
            used_notes = []

            def replace_note_link(match):
                nonlocal footnote_counter
                preceding_text = match.group(1)
                note_id = match.group(2)

                if note_id not in note_map:
                    return match.group(0)

                used_notes.append(note_id)

                if preceding_text:
                    label = preceding_text.strip()
                else:
                    label = f"[{footnote_counter}]"
                    footnote_counter += 1

                return f'<a epub:type="noteref" href="#{note_id}" class="footnote-link">{label}</a>'

            pattern = re.compile(r"(\(\d+\)|\[\d+\])?\s*\[(note\d+)\]")
            html_content = pattern.sub(replace_note_link, html_content)

            footnotes_html = ""

            def create_footnote_block(nid, content, title="Ghi chú"):
                return f'''
                <aside id="{nid}" epub:type="footnote" class="footnote-content">
                    <div class="note-header">{title}:</div>
                    <p>{content}</p>
                </aside>
                '''

            for nid in used_notes:
                content = note_map.get(nid, "")
                footnotes_html += create_footnote_block(nid, content, "Ghi chú")

            for nid, content in note_map.items():
                if nid not in used_notes:
                    footnotes_html += create_footnote_block(
                        nid, content, "Ghi chú (Thêm)"
                    )

            final_html = html_content + footnotes_html

            return {
                "title": chapter.name,
                "url": chapter.url,
                "content": final_html,
                "index": idx,
            }
        except Exception as e:
            logger.error(f"Err {chapter.url}: {e}")
            return None

    def _validate_cached_chapter(self, chapter_data: Dict) -> bool:
        if not chapter_data or "content" not in chapter_data:
            return False
        if len(chapter_data["content"]) < 50:
            return False
        try:
            soup = BeautifulSoup(chapter_data["content"], HTML_PARSER)
            images = soup.find_all("img")
            for img in images:
                src = img.get("src")
                if src and src.startswith("images/"):
                    full_path = join(self.base_folder, src)
                    if not exists(full_path) or os.path.getsize(full_path) == 0:
                        return False
        except Exception:
            return False
        return True

    def download_volume(self, volume: Volume):
        json_filename = TextUtils.format_filename(volume.name) + ".json"
        json_path = join(self.base_folder, json_filename)
        vol_slug = TextUtils.format_filename(volume.name).lower()

        existing_chapters = {}
        if exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    for ch in old_data.get("chapters", []):
                        existing_chapters[ch["url"]] = ch
            except Exception:
                logger.warning("Existing JSON corrupt.")

        tasks = []
        final_chapters = []

        print(f"Processing Volume: {volume.name}")

        cached_count = 0
        re_download_count = 0

        for i, chap in enumerate(volume.chapters):
            cached_data = existing_chapters.get(chap.url)
            if cached_data and self._validate_cached_chapter(cached_data):
                cached_data["index"] = i
                final_chapters.append(cached_data)
                cached_count += 1
            else:
                if cached_data:
                    re_download_count += 1
                tasks.append((i, chap, vol_slug))

        print(f"Cached: {cached_count} | Re-downloading: {len(tasks)}")

        if tasks:
            pool = ThreadPool(THREAD_NUM)
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(self._process_chapter, tasks), total=len(tasks)
                )
            )
            pool.close()
            pool.join()
            for res in results:
                if res:
                    final_chapters.append(res)

        final_chapters.sort(key=lambda x: x["index"])

        vol_cover_local = ""
        if volume.cover_img:
            ext = "jpg"
            if "png" in volume.cover_img:
                ext = "png"
            elif "gif" in volume.cover_img:
                ext = "gif"

            fname = f"vol_cover_{TextUtils.format_filename(volume.name)}.{ext}"
            if NetworkManager.download_image_to_disk(
                volume.cover_img, join(self.images_folder, fname)
            ):
                vol_cover_local = f"images/{fname}"

        volume_data = {
            "volume_name": volume.name,
            "volume_url": volume.url,
            "cover_image_local": vol_cover_local,
            "chapters": final_chapters,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(volume_data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {json_path}")


# --- Phase 2: EPUB Builder ---


class EpubBuilder:
    def __init__(self, base_folder: str, compress_images: bool = True):
        self.base_folder = base_folder
        self.compress_images = compress_images
        self.image_map = {}
        self.result_root = RESULT_DIR

        self.meta = {}
        meta_path = join(base_folder, "metadata.json")
        if exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {
                "novel_name": "Unknown",
                "author": "Unknown",
                "summary": "",
                "cover_image_local": "",
                "tags": [],
            }

        self.css = """
            body { margin: 0; padding: 5px; text-align: justify; line-height: 1.4em; font-family: serif; }
            h1, h2, h3 { text-align: center; margin: 1em 0; font-weight: bold; }
            img { display: block; margin: 10px auto; max-width: 100%; height: auto; }
            p { margin-bottom: 1em; text-indent: 1em; }
            .center { text-align: center; }
            nav#toc ol { list-style-type: none; padding-left: 0; }
            nav#toc > ol > li { margin-top: 1em; font-weight: bold; }
            nav#toc > ol > li > ol { list-style-type: none; padding-left: 1.5em; font-weight: normal; }
            nav#toc > ol > li > ol > li { margin-top: 0.5em; }
            nav#toc a { text-decoration: none; color: inherit; }
            nav#toc a:hover { text-decoration: underline; color: blue; }
            a.footnote-link { vertical-align: super; font-size: 0.75em; text-decoration: none; color: #007bff; margin-left: 2px; }
            aside.footnote-content { margin-top: 1em; padding: 0.5em; border-top: 1px solid #ccc; font-size: 0.9em; color: #333; background-color: #f9f9f9; display: block; }
            aside.footnote-content p { margin: 0; text-indent: 0; }
            aside.footnote-content div.note-header { font-weight: bold; margin-bottom: 0.5em; color: #555; }
        """

    def _get_output_path(self, filename: str, is_merged: bool) -> str:
        """
        Determines the final output path based on user rules.
        1. Merged + Original -> result/<BookName - Full>.epub
        2. Compressed (Any) -> result/<BookName>/compressed/<filename>
        3. Separate + Original -> result/<BookName>/original/<filename>
        """
        book_name_slug = TextUtils.format_filename(self.meta["novel_name"])

        # Case 1: Merged & Original (Special Case)
        if is_merged and not self.compress_images:
            if not exists(self.result_root):
                os.makedirs(self.result_root)
            return join(self.result_root, filename)

        # Determine subfolder
        subfolder = "compressed" if self.compress_images else "original"

        # Path: result/<BookName>/<subfolder>/
        target_dir = join(self.result_root, book_name_slug, subfolder)

        if not exists(target_dir):
            os.makedirs(target_dir)

        return join(target_dir, filename)

    def sanitize_xhtml(self, html_content: str) -> str:
        if not html_content:
            return ""
        safe = html_content
        safe = safe.replace("&nbsp;", "&#160;")
        pattern_empty_p = re.compile(
            r"<p[^>]*>(\s|&nbsp;|&#160;|<br\s*\/?>)*<\/p>", re.IGNORECASE
        )
        safe = pattern_empty_p.sub("", safe)
        pattern_br = re.compile(r"(<br\s*\/?>\s*){3,}", re.IGNORECASE)
        safe = pattern_br.sub("<br/><br/>", safe)
        return safe.strip()

    def process_image(self, rel_path: str) -> Tuple[Optional[epub.EpubItem], str]:
        if not rel_path:
            return None, ""

        if rel_path in self.image_map:
            return None, self.image_map[rel_path]

        full_path = join(self.base_folder, rel_path)
        if not exists(full_path):
            return None, rel_path

        try:
            with Image.open(full_path) as valid_check:
                valid_check.load()

            if not self.compress_images:
                with open(full_path, "rb") as f:
                    content = f.read()

                ext = splitext(rel_path)[1].lower()
                media_type = "image/jpeg"
                if ext == ".png":
                    media_type = "image/png"
                elif ext == ".gif":
                    media_type = "image/gif"
                elif ext == ".webp":
                    media_type = "image/webp"

                item = epub.EpubItem(
                    uid=rel_path.replace("/", "_").replace(".", "_"),
                    file_name=rel_path,
                    media_type=media_type,
                    content=content,
                )
                self.image_map[rel_path] = rel_path
                return item, rel_path

            img = Image.open(full_path)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            output = io.BytesIO()
            img.save(output, format="JPEG", quality=75, optimize=True)

            base, _ = splitext(rel_path)
            new_rel_path = f"{base}.jpg"

            item = epub.EpubItem(
                uid=new_rel_path.replace("/", "_").replace(".", "_"),
                file_name=new_rel_path,
                media_type="image/jpeg",
                content=output.getvalue(),
            )

            self.image_map[rel_path] = new_rel_path
            return item, new_rel_path

        except Exception as e:
            error_msg = str(e).lower()
            if "truncated" in error_msg or "cannot identify" in error_msg:
                logger.error(f"CORRUPT FILE FOUND: {rel_path}")
                logger.warning(f"-> Deleting {rel_path}...")
                try:
                    os.remove(full_path)
                    logger.warning(
                        "-> File deleted. Please run 'Download' action again to re-fetch it."
                    )
                except OSError:
                    logger.error("-> Could not delete file (locked?). Delete manually.")
                return None, ""

            logger.error(f"Image process failed for {rel_path}: {e}")
            return None, rel_path

    def make_intro(self, vol_name: str = ""):
        summary_html = self.sanitize_xhtml(self.meta.get("summary", ""))
        title = html.escape(self.meta["novel_name"])
        author = html.escape(self.meta["author"])
        tags_str = ", ".join(self.meta.get("tags", []))
        tags_html = f"<p><b>Thể loại:</b> {tags_str}</p>" if tags_str else ""

        cover_html = "<hr/>"
        main_cover_path = self.meta.get("cover_image_local")

        cover_item = None
        if main_cover_path:
            item, new_path = self.process_image(main_cover_path)
            cover_item = item
            if new_path:
                cover_html = f'<div style="text-align:center; margin: 2em 0; page-break-after: always; break-after: page;"><img src="{new_path}" alt="Cover"/></div>'

        content = f"""
            <div style="text-align: center; margin-top: 5%;">
                <h1>{title}</h1>
                <h3 style="margin-bottom: 0.5em;">{vol_name}</h3>
                <p><b>Tác giả:</b> {author}</p>
                {tags_html}
                {cover_html}
                <div style="text-align: justify;">
                    {summary_html}
                </div>
            </div>
        """
        page = epub.EpubHtml(
            title="Giới thiệu", file_name="intro.xhtml", content=content
        )
        return page, cover_item

    def build_merged(self, json_files: List[str]):
        if "volumes" in self.meta:
            order_map = {v["filename"]: v["order"] for v in self.meta["volumes"]}
            json_files.sort(key=lambda x: order_map.get(x, 9999))

        book = epub.EpubBook()
        book.set_title(f"{self.meta['novel_name']}")
        book.set_language("vi")
        book.add_author(self.meta["author"])

        if self.meta.get("summary"):
            book.add_metadata("DC", "description", self.meta["summary"])
        for tag in self.meta.get("tags", []):
            book.add_metadata("DC", "subject", tag)

        css = epub.EpubItem(
            uid="style", file_name="style.css", media_type="text/css", content=self.css
        )
        book.add_item(css)

        intro_page, main_cover_item = self.make_intro("Toàn tập")
        intro_page.add_item(css)

        if main_cover_item:
            book.add_item(main_cover_item)
            book.set_cover("cover.jpg", main_cover_item.content, create_page=False)

        book.add_item(intro_page)
        spine = [intro_page]
        toc = [epub.Link("intro.xhtml", "Giới thiệu", "intro")]

        for i, jf in enumerate(json_files):
            print(f"Merging: {jf}")
            with open(join(self.base_folder, jf), "r", encoding="utf-8") as f:
                vol_data = json.load(f)

            vol_html_content = ""
            vol_cover = vol_data.get("cover_image_local")

            if vol_cover:
                item, new_src = self.process_image(vol_cover)
                if item:
                    book.add_item(item)
                vol_html_content += (
                    f'<img src="{new_src}" alt="Vol Cover" style="max-height: 50vh;"/>'
                )

            vol_html_content += f"<h1>{html.escape(vol_data['volume_name'])}</h1>"

            sep_html = f"""
                <div style="text-align: center; margin-top: 30vh;">
                    {vol_html_content}
                </div>
            """

            sep_page = epub.EpubHtml(
                title=vol_data["volume_name"],
                file_name=f"vol_{i}.xhtml",
                content=sep_html,
            )
            sep_page.add_item(css)
            book.add_item(sep_page)
            spine.append(sep_page)

            vol_chaps = []
            for chap in vol_data["chapters"]:
                soup = BeautifulSoup(chap["content"], HTML_PARSER)
                for img in soup.find_all("img"):
                    src = img.get("src")
                    if src:
                        item, new_src = self.process_image(src)
                        if item:
                            book.add_item(item)
                        img["src"] = new_src

                clean_content = self.sanitize_xhtml(str(soup))
                fname = f"v{i}_c{chap['index']}.xhtml"
                c_page = epub.EpubHtml(
                    title=chap["title"],
                    file_name=fname,
                    content=f"<h2>{html.escape(chap['title'])}</h2>{clean_content}",
                )
                c_page.add_item(css)
                book.add_item(c_page)
                vol_chaps.append(c_page)

            spine.extend(vol_chaps)
            toc.append((sep_page, vol_chaps))

        nav = epub.EpubNav()
        book.add_item(nav)
        book.spine = ["nav"] + spine
        book.toc = toc
        book.add_item(epub.EpubNcx())

        filename = TextUtils.format_filename(f"{self.meta['novel_name']} Full.epub")
        out_path = self._get_output_path(filename, is_merged=True)

        epub.write_epub(out_path, book, {})
        print(f"Created Merged EPUB: {out_path}")

    def build_volume(self, json_file: str):
        with open(join(self.base_folder, json_file), "r", encoding="utf-8") as f:
            vol_data = json.load(f)

        book = epub.EpubBook()
        book.set_title(f"{vol_data['volume_name']} - {self.meta['novel_name']}")
        book.set_language("vi")
        book.add_author(self.meta["author"])
        for tag in self.meta.get("tags", []):
            book.add_metadata("DC", "subject", tag)

        css = epub.EpubItem(
            uid="style", file_name="style.css", media_type="text/css", content=self.css
        )
        book.add_item(css)

        intro_page, main_cover_item = self.make_intro(vol_data["volume_name"])
        intro_page.add_item(css)
        if main_cover_item:
            book.add_item(main_cover_item)
            book.set_cover("cover.jpg", main_cover_item.content, create_page=False)
        book.add_item(intro_page)

        spine = [intro_page]

        for chap in vol_data["chapters"]:
            soup = BeautifulSoup(chap["content"], HTML_PARSER)
            for img in soup.find_all("img"):
                src = img.get("src")
                if src:
                    item, new_src = self.process_image(src)
                    if item:
                        book.add_item(item)
                    img["src"] = new_src

            clean = self.sanitize_xhtml(str(soup))
            c_page = epub.EpubHtml(
                title=chap["title"],
                file_name=f"c{chap['index']}.xhtml",
                content=f"<h2>{html.escape(chap['title'])}</h2>{clean}",
            )
            c_page.add_item(css)
            book.add_item(c_page)
            spine.append(c_page)

        nav = epub.EpubNav()
        book.add_item(nav)
        book.spine = ["nav"] + spine
        book.add_item(epub.EpubNcx())

        filename = TextUtils.format_filename(
            f"{vol_data['volume_name']} - {self.meta['novel_name']}.epub"
        )
        out_path = self._get_output_path(filename, is_merged=False)

        epub.write_epub(out_path, book, {})
        print(f"Created: {out_path}")


# --- Parser ---


class LightNovelInfoParser:
    def parse(self, url: str) -> Optional[LightNovel]:
        print("Fetching Novel Info...", end="\r")
        try:
            resp = NetworkManager.check_available_request(url)
            soup = BeautifulSoup(resp.text, HTML_PARSER)
            ln = LightNovel(url=url)
            name_tag = soup.find("span", "series-name")
            ln.name = name_tag.text.strip() if name_tag else "Unknown"

            cover_div = soup.find("div", "series-cover")
            if cover_div:
                img_div = cover_div.find("div", "img-in-ratio")
                if img_div and "style" in img_div.attrs:
                    match = re.search(
                        r'url\([\'"]?([^\'"\)]+)[\'"]?\)', img_div["style"]
                    )
                    if match:
                        ln.main_cover = match.group(1)

            # Get Tags
            genre_div = soup.find(
                "div", class_=re.compile(r"series-gernes|series-genres")
            )
            if genre_div:
                for a in genre_div.find_all("a"):
                    ln.tags.append(a.text.strip())

            info_div = soup.find("div", "series-information")
            if info_div:
                for item in info_div.find_all("div", "info-item"):
                    label = item.find("span", "info-name")
                    if label and "Tác giả" in label.text:
                        val = item.find("span", "info-value")
                        if val:
                            ln.author = val.text.strip()

            sum_div = soup.find("div", "summary-content")
            if sum_div:
                for bad in sum_div.find_all(
                    ["a", "div", "span"],
                    class_=["see-more", "less-state", "more-state"],
                ):
                    bad.decompose()
                ln.summary = "".join([str(x) for x in sum_div.contents]).strip()

            for sect in soup.find_all("section", "volume-list"):
                vol = Volume()
                title = sect.find("span", "sect-title")
                vol.name = title.text.strip() if title else "Unknown Vol"

                v_cover = sect.find("div", "volume-cover")
                if v_cover:
                    a = v_cover.find("a")
                    if a:
                        vol.url = TextUtils.reformat_url(url, a["href"])
                    img = v_cover.find("div", "img-in-ratio")
                    if img and "style" in img.attrs:
                        match = re.search(
                            r'url\([\'"]?([^\'"\)]+)[\'"]?\)', img["style"]
                        )
                        if match:
                            vol.cover_img = match.group(1)

                ul = sect.find("ul", "list-chapters")
                if ul:
                    for li in ul.find_all("li"):
                        a = li.find("a")
                        if a:
                            c_url = TextUtils.reformat_url(url, a["href"])
                            vol.chapters.append(Chapter(name=a.text.strip(), url=c_url))
                ln.volumes.append(vol)

            print(f"Parsed: {ln.name} | Tags: {len(ln.tags)}")
            return ln
        except Exception as e:
            logger.error(f"Parse Error: {e}")
            return None


# --- Application ---


class Application:
    def __init__(self, url=None):
        self.parser = LightNovelInfoParser()
        self.cli_url = url

    def run(self):
        action = questionary.select(
            "Select Action:",
            choices=[
                "Download (Create JSONs)",
                "Build EPUB (From JSONs)",
                "Full Process",
                "Batch Build (All Books, All Options)",
            ],
        ).ask()

        # --- BATCH PROCESSING ---
        if action == "Batch Build (All Books, All Options)":
            if not exists(DATA_DIR):
                print(f"No {DATA_DIR} directory found.")
                return

            books = [f for f in os.listdir(DATA_DIR) if isdir(join(DATA_DIR, f))]
            if not books:
                print("No books found in data directory.")
                return

            print(f"Found {len(books)} books. Starting batch process...")

            for book_folder in books:
                book_path = join(DATA_DIR, book_folder)
                jsons = [
                    f
                    for f in os.listdir(book_path)
                    if f.endswith(".json") and f != "metadata.json"
                ]

                if not jsons:
                    continue

                print(f"\n>>> PROCESSING BOOK: {book_folder}")

                # Loop through compression modes: [Optimized(True), Original(False)]
                for compress_mode in [True, False]:
                    mode_name = "Optimized" if compress_mode else "Original"
                    print(f"   > Building Mode: {mode_name}")

                    # Must re-init builder to clear image_map cache between modes
                    builder = EpubBuilder(book_path, compress_images=compress_mode)

                    # 1. Build Merged
                    builder.build_merged(jsons)

                    # 2. Build Separate
                    # Sort separate volumes for cleaner logs (optional but nice)
                    # We rely on metadata order if possible, else filename
                    if "volumes" in builder.meta:
                        order_map = {
                            v["filename"]: v["order"] for v in builder.meta["volumes"]
                        }
                        jsons.sort(key=lambda x: order_map.get(x, 9999))

                    for j in jsons:
                        builder.build_volume(j)

            print("\nBatch process finished!")
            return
        # ------------------------

        url = ""
        ln = None
        save_dir = ""

        if action != "Build EPUB (From JSONs)":
            if self.cli_url:
                url = self.cli_url
            else:
                url = questionary.text("Light Novel URL:").ask()
            ln = self.parser.parse(url)
            if not ln:
                return
            save_dir = join(DATA_DIR, TextUtils.format_filename(ln.name))
        else:
            if not exists(DATA_DIR):
                print(f"No {DATA_DIR} directory found.")
                return
            folders = [f for f in os.listdir(DATA_DIR) if isdir(join(DATA_DIR, f))]
            if not folders:
                return
            fname = questionary.select("Select Folder:", choices=folders).ask()
            save_dir = join(DATA_DIR, fname)

        if action in ["Download (Create JSONs)", "Full Process"]:
            dl = NovelDownloader(ln, save_dir)
            dl.create_metadata_file()

            opts = [v.name for v in ln.volumes]
            sel = questionary.checkbox("Select Volumes:", choices=opts).ask()
            if not sel:
                return

            targets = (
                ln.volumes
                if "All Volumes" in sel
                else [v for v in ln.volumes if v.name in sel]
            )

            for v in targets:
                dl.download_volume(v)

        if action in ["Build EPUB (From JSONs)", "Full Process"]:
            compress_choice = questionary.select(
                "Image Quality for EPUB:",
                choices=[
                    "Optimized (Small File Size - JPEG 75)",
                    "Original (Max Quality - Keep PNG/GIF)",
                ],
            ).ask()

            do_compress = "Optimized" in compress_choice

            builder = EpubBuilder(save_dir, compress_images=do_compress)
            jsons = [
                f
                for f in os.listdir(save_dir)
                if f.endswith(".json") and f != "metadata.json"
            ]
            if not jsons:
                print("No volume JSONs found.")
                return

            btype = questionary.select(
                "Mode:", choices=["Separate EPUBs", "Merged EPUB"]
            ).ask()
            sel_jsons = questionary.checkbox("Select Volumes:", choices=jsons).ask()

            if not sel_jsons:
                return

            if btype == "Separate EPUBs":
                for j in sel_jsons:
                    builder.build_volume(j)
            else:
                builder.build_merged(sel_jsons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hako2Epub Downloader")
    parser.add_argument("url", nargs="?", help="The URL of the light novel to download")
    args = parser.parse_args()

    try:
        app = Application(args.url)
        app.run()
    except KeyboardInterrupt:
        print("\nExit.")
