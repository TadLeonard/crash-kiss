#!/usr/bin/env python3

import argparse
import time
import traceback

from subprocess import run as run_process
from tfatool.sync import watch_local_files


parser = argparse.ArgumentParser()
parser.add_argument("print_dir", help="Print new files that appear "
                                      "in PRINT_DIR")


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
            print(". . {}\r".format(busy), end="")
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
    while True:
        try:
            main(args)
        except KeyboardInterrupt:
            print("Bye!")
            break
        except Exception:
            traceback.print_exc()
            print("Sleeping five seconds after error")
            time.sleep(5)

