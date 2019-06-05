"""
A small DSL for defining directory structures for use in testing directory
comparison.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Optional


class Entry(ABC):
    @abstractmethod
    def instantiate(self, path: Path) -> None:
        ...


class Dir(Entry):
    def __init__(self, entries: Optional[Mapping[str, Entry]] = None):
        if entries is None:
            entries = dict()
        self.entries = entries

    def instantiate(self, path: Path):
        path.mkdir()

        for name, entry in self.entries.items():
            entry.instantiate(path / name)

    def __repr__(self) -> str:
        return f"Dir({repr(self.entries)})"


class File(Entry):
    def __init__(self, content: str = ""):
        self.content = content

    def instantiate(self, path: Path):
        path.write_text(self.content)

    def __repr__(self) -> str:
        return f"File({repr(self.content)})"
