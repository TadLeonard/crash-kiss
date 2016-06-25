#!/usr/bin/env python3

import argparse
import time
import traceback
import pathlib

from subprocess import run as run_process
from tfatool.sync import watch_local_files


parser = argparse.ArgumentParser()
parser.add_argument("print_dir", default=".", nargs="?",
                    help="Print new files that appear in PRINT_DIR")
parser.add_argument("--vernal-pond", action="store_true",
                    help="use default VLC screen cap directory for "
                         "CMCA Vernal Pond show")


_print = print


def print(to_print, *args, **kwargs):
    _print("[printd] {}".format(to_print), *args, **kwargs)


def main(args):
    busy = "."
    is_jpg = lambda f: f.filename.lower().endswith(".jpg")
    watcher = watch_local_files(is_jpg, local_dir=args.print_dir)
    for new_files, _ in watcher:
        time.sleep(0.2)
        if not new_files:
            busy = " " if busy == "." else "."
            print("Waiting for files in {}. . {}\r".format(
                  args.print_dir, busy), end="")
            continue
        for new_file in new_files:
            print_file(new_file)
            break


def print_file(new_file):
    print("Printing {}".format(new_file.path))
    print("Waiting three seconds to ensure the file has been written")
    time.sleep(3)
    try:
        run_process(["selphy", new_file.path], timeout=20)
    except OSError:
        traceback.print_exc()
        print("Sleeping two seconds after file printing error")
        time.sleep(2)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.vernal_pond:
        path = pathlib.Path("~/Pictures/vernal_pond/prints/").expanduser()
        args.print_dir = str(path)
    while True:
        try:
            main(args)
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception:
            traceback.print_exc()
        print("Sleeping five seconds after error")
        time.sleep(5)

