import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from .constants import HTML_PARSER
from .models import Chapter, LightNovel, Volume
from .network import NetworkManager
from .utils import TextUtils

logger = logging.getLogger(__name__)


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
                "div", class_=re.compile(r"series-gernes|series-genres"))
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
                    class_=[ "see-more", "less-state", "more-state"],
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
                            vol.chapters.append(
                                Chapter(name=a.text.strip(), url=c_url)
                            )
                ln.volumes.append(vol)

            print(f"Parsed: {ln.name} | Tags: {len(ln.tags)}")
            return ln
        except Exception as e:
            logger.error(f"Parse Error: {e}")
            return None
