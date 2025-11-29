import logging
import os
import json
from os.path import isdir, join, exists

import questionary

from .constants import BOOKS_FILE, DATA_DIR
from .downloader import NovelDownloader
from .epub_builder import EpubBuilder
from .parser import LightNovelInfoParser
from .utils import TextUtils, read_books_list

logger = logging.getLogger(__name__)


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
                "Batch Build (from books.json)",
            ],
        ).ask()

        # --- BATCH PROCESSING ---
        if action == "Batch Build (from books.json)":
            books_in_list = read_books_list()
            if not books_in_list:
                print(f"'{BOOKS_FILE}' is empty. Nothing to build.")
                return

            print(
                f"Found {len(books_in_list)} books in '{BOOKS_FILE}'. Starting batch process..."
            )

            for book_folder in books_in_list:
                book_path = join(DATA_DIR, book_folder)
                if not isdir(book_path):
                    logger.warning(
                        f"Book folder '{book_folder}' not found in '{DATA_DIR}'. Skipping."
                    )
                    continue

                jsons = [
                    f
                    for f in os.listdir(book_path)
                    if f.endswith(".json") and f != "metadata.json"
                ]

                if not jsons:
                    logger.warning(
                        f"No volume jsons found for '{book_folder}'. Skipping."
                    )
                    continue

                print(f"\n>>> PROCESSING BOOK: {book_folder}")

                # Loop through compression modes: [Optimized(True), Original(False)]
                for compress_mode in [True, False]:
                    mode_name = "Optimized" if compress_mode else "Original"
                    print(f"   > Building Mode: {mode_name}")

                    # Must re-init builder to clear image_map cache between modes
                    builder = EpubBuilder(book_path, compress_images=compress_mode)

                    # 1. Build Merged
                    base_merged_filename = TextUtils.format_filename(
                        f"{builder.meta['novel_name']} Full"
                    )
                    merged_filename = f"{base_merged_filename}.epub"
                    merged_out_path = builder._get_output_path(
                        merged_filename, is_merged=True, create_dirs=False
                    )

                    if exists(merged_out_path):
                        print("   - Merged EPUB already exists, skipping.")
                    else:
                        print("   - Building Merged EPUB...")
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
                        try:
                            with open(
                                join(book_path, j), "r", encoding="utf-8"
                            ) as f:
                                vol_data = json.load(f)

                            base_sep_filename = TextUtils.format_filename(
                                f"{vol_data['volume_name']} - {builder.meta['novel_name']}"
                            )
                            sep_filename = f"{base_sep_filename}.epub"
                            sep_out_path = builder._get_output_path(
                                sep_filename, is_merged=False, create_dirs=False
                            )

                            if exists(sep_out_path):
                                print(
                                    f"   - EPUB for '{vol_data['volume_name']}' already exists, skipping."
                                )
                            else:
                                print(
                                    f"   - Building EPUB for '{vol_data['volume_name']}'..."
                                )
                                builder.build_volume(j)
                        except Exception as e:
                            logger.error(
                                f"Failed to process volume '{j}' for '{book_folder}': {e}"
                            )

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
