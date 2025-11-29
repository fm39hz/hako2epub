import json
import logging
import os
import re
import time
from multiprocessing.dummy import Pool as ThreadPool
from os.path import exists, join
from typing import Dict, Tuple

import tqdm
from bs4 import BeautifulSoup

from .constants import HTML_PARSER, THREAD_NUM
from .models import Chapter, LightNovel, Volume
from .network import NetworkManager
from .utils import TextUtils, add_book_to_list

logger = logging.getLogger(__name__)


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

        book_folder_name = os.path.basename(self.base_folder)
        add_book_to_list(book_folder_name)

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
                return f"""
                <aside id="{nid}" epub:type="footnote" class="footnote-content">
                    <div class="note-header">{title}:</div>
                    <p>{content}</p>
                </aside>
                """

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
                    pool.imap_unordered(self._process_chapter, tasks),
                    total=len(tasks),
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
