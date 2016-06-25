#!/usr/bin/env python3

import os
import shutil
import time
import traceback

from kiss import parser, run_animate
from tfatool import sync


booth_group = parser.add_argument_group("booth options")
booth_group.add_argument("--photo-input-dir", default=".")
booth_group.add_argument("--crash-output-dir", default=".")
booth_group.add_argument("--crash-file", default="crash.mp4")


_print = print


def print(to_print, *args, **kwargs):
    _print("[crash] {}".format(to_print), *args, **kwargs)


def main(args):
    only_jpg = lambda path: path.filename.lower().endswith(".jpg")
    watcher = sync.down_by_arrival(only_jpg, local_dir=args.photo_input_dir)
    try:
        create_animations(args, watcher)
    except KeyboardInterrupt:
        print("Bye!")


def create_animations(args, watcher):
    print("Watching for new photos in {}".format(args.photo_input_dir))
    busy = "."
    for _, new_photos in watcher:
        if not new_photos:
            busy = " " if busy == "." else "."
            print(". . {}\r".format(busy), end="")
            time.sleep(0.2)
        else:
            for photo in new_photos:
                local_file = os.path.join(args.photo_input_dir, photo.filename)
                print("Crashing photo: {} from {}".format(local_file, photo.path))
                run_animate(args, local_file, args.crash_file)
                outname = photo.filename.split(".")[0]
                outext = args.crash_file.split(".")[-1]
                outfile = ".".join([outname, outext])
                crash_copy = os.path.join(args.crash_output_dir, outfile)
                print("Copying {} to {}".format(args.crash_file, crash_copy))
                shutil.copyfile(args.crash_file, crash_copy)


if __name__ == "__main__":
    args = parser.parse_args()
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

