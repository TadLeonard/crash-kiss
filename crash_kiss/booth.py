import os

from collections import deque
from pathlib import Path
from typing import Tuple, Callable, Iterator, Optional, Set, Deque


FileFilter = Callable[(Path,), bool]


def iter_files(
    *filters: Tuple[FileFilter, ...], directory: Optional[Path] = None
) -> Iterator[Path]:
    if directory is None:
        directory = Path.cwd()
    file_entries = (e for e in directory.iterdir() if e.is_file())
    for entry in file_entries:
        if all(filt(entry) for filt in filters):
            yield entry


def _is_jpg(path: Path) -> bool:
    return path.name.lower().endswith(("jpg", "jpeg"))


class NewJpgs:
    """
    Generates `Path` objects pointing to newly created JPEG files.

    Keeps track of consumed (yielded, iterated over) paths with
    `consumed` in the order they were consumed. New, unconsumed files are
    tracked with `new` and they're ordered by `Path.stat().st_mtime`.

    The filesystem may be checked for new files on any iteration. NewJpgs
    is an iterator that can be exhausted repeatedly.
    """

    path: Path
    _new: Deque[Path]
    _consumed: Deque[Path]
    _initial_contents: Set[Path]

    def __init__(self, path: Path):
        self._new = deque()
        self._consumed = deque()
        self.path = path
        self._initial_contents = set(self._list_paths())

    def _list_paths(self) -> Iterator[Path]:
        return iter_files(_is_jpg, directory=self.path)

    @property
    def new(self) -> Deque[Path]:
        return deque(self._new)

    @property
    def consumed(self) -> Deque[Path]:
        return deque(self._consumed)

    def scan(self) -> bool:
        """Returns True if there are unconsumed files, False otherwise"""
        new_paths = [
            path
            for path in self._list_paths()
            if path not in self._initial_contents
            and path not in self._consumed
            and path not in self._new
        ]
        old_to_new = sorted(new_paths, key=lambda p: p.stat().st_mtime)
        self._new.extend(old_to_new)
        return bool(self._new)

    def __iter__(self):
        while True:
            new_item = next(self)
            if new_item is None:
                break
            else:
                yield new_item

    def __next__(self):
        if self._new or self.scan():
            new_jpg = self._new.popleft()
            self._consumed.append(new_jpg)
            return new_jpg
