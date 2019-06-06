"""
A small DSL for defining directory structures for use in testing directory
comparison.

"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


class Entry(ABC):
    @abstractmethod
    def instantiate(self, path: Path) -> None:
        ...


@dataclass
class Dir(Entry):
    entries: Mapping[str, Entry] = field(default_factory=dict)

    def instantiate(self, path: Path):
        path.mkdir()

        for name, entry in self.entries.items():
            entry.instantiate(path / name)


@dataclass
class File(Entry):
    content: str = ""

    def instantiate(self, path: Path):
        path.write_text(self.content)
