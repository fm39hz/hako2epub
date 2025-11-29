import html
import io
import json
import logging
import os
import re
from os.path import exists, join, splitext
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup
from ebooklib import epub
from PIL import Image, ImageFile

from .constants import RESULT_DIR
from .utils import TextUtils

# --- FIX FOR TRUNCATED IMAGES ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
# -----

logger = logging.getLogger(__name__)


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

    def _get_output_path(
        self, filename: str, is_merged: bool, create_dirs: bool = True
    ) -> str:
        """
        Determines the final output path based on user rules.
        1. Merged + Original -> result/<BookName - Full>.epub
        2. Compressed (Any) -> result/<BookName>/compressed/<filename>
        3. Separate + Original -> result/<BookName>/original/<filename>
        """
        book_name_slug = TextUtils.format_filename(self.meta["novel_name"])

        # Case 1: Merged & Original (Special Case)
        if is_merged and not self.compress_images:
            if create_dirs and not exists(self.result_root):
                os.makedirs(self.result_root)
            return join(self.result_root, filename)

        # Determine subfolder
        subfolder = "compressed" if self.compress_images else "original"

        # Path: result/<BookName>/<subfolder>/
        target_dir = join(self.result_root, book_name_slug, subfolder)

        if create_dirs and not exists(target_dir):
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
                soup = BeautifulSoup(chap["content"], "html.parser")
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

        base_filename = TextUtils.format_filename(f"{self.meta['novel_name']} Full")
        filename = f"{base_filename}.epub"
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
            soup = BeautifulSoup(chap["content"], "html.parser")
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

        base_filename = TextUtils.format_filename(
            f"{vol_data['volume_name']} - {self.meta['novel_name']}"
        )
        filename = f"{base_filename}.epub"
        out_path = self._get_output_path(filename, is_merged=False)

        epub.write_epub(out_path, book, {})
        print(f"Created: {out_path}")
