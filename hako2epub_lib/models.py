from dataclasses import dataclass, field
from typing import List


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
