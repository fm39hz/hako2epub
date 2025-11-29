import logging
import os
import shutil
import time
from urllib.parse import urlparse

import requests

from .constants import DOMAINS, HEADERS, IMAGE_DOMAINS

logger = logging.getLogger(__name__)
session = requests.Session()


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
            logger.info("Anti-Ban: Pausing for 30 seconds...")
            time.sleep(30)

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

        if os.path.exists(save_path):
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
