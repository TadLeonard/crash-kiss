#!/usr/bin/env python3

import argparse
import os
import time

from tfatool.sync import watch_local_files


parser = argparse.ArgumentParser()
parser.add_argument("print_dir")

args = parser.parse_args()
busy = "."
for new_files, _ in watch_local_files(local_dir=args.print_dir):
    time.sleep(0.2)
    if not new_files:
        busy = " " if busy == "." else "."
        print(". . {}\r".format(busy), end="")
        continue
    for new_file in new_files:
        print("Printing {}".format(new_file.path))
        time.sleep(1)
        os.system("selphy {}".format(new_file.path))
        time.sleep(5)
        break

