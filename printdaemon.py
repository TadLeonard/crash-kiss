#!/usr/bin/env python3

import argparse
import os
import time
import traceback

from tfatool.sync import watch_local_files


parser = argparse.ArgumentParser()
parser.add_argument("print_dir")
args = parser.parse_args()


_print = print


def print(to_print, *args, **kwargs):
    _print("[printd] {}".format(to_print), *args, **kwargs)


def main():
    busy = "."
    for new_files, _ in watch_local_files(local_dir=args.print_dir):
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
        os.system("selphy {}".format(new_file.path))
    except OSError:
        traceback.print_exc()
        print("Sleeping five seconds after file printing error")
        time.sleep(5)



if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("Bye!")
            break
        except Exception:
            traceback.print_exc()
            print("Sleeping five seconds after error")
            time.sleep(5)

