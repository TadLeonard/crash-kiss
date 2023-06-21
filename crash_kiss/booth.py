import os

from pathlib import Path
from typing import Tuple, Callable, Iterator, Optional


FileFilter = Callable[(Path,), bool]


def local_files(
    *filters: Tuple[FileFilter, ...], directory: Optional[Path] = None
) -> Iterator[Path]:
    if directory is None:
        directory = Path.cwd()
    file_entries = (e for e in directory.iterdir() if e.is_file())
    for entry in file_entries:
        if all(filt(entry) for filt in filters):
            yield entry
