#!/usr/bin/env python
"""
Smashes the things on the left and right side of an image towards the center
"""

import argparse
from crash_kiss import edge, mutate, config
import imread


_conf = config.config()   # default conf values
parser = argparse.ArgumentParser(
	description="Crash two faces into each other",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("target")
parser.add_argument("-b", "--bg-value", type=lambda x: map(int, x.split(",")),
                      help="A number to represent the color of the background "
                           "should the user want to manually set it. Use "
                           "'auto' to automatically gather per-row "
                           "background values.",
                      default=_conf["bg_value"])
parser.add_argument("-o", "--outfile", default=None)
parser.add_argument("-e", "--reveal-foreground", action="store_true")
parser.add_argument("-E", "--reveal-background", action="store_true")


def main():
    args = parser.parse_args()
    img = imread.imread(args.target)
    conf = config.config(bg_value=args.bg_value)
    bg = conf["bg_value"]  # no dynamic background gathering
    fg = edge.find_foreground(img, bg, conf)
     
    # Various things to do with the result of our image mutations
    if args.reveal_foreground:
        mutate.reveal_foreground(img, fg)
    if args.reveal_background:
        mutate.reveal_background(img, fg)
    if args.outfile:
        imread.imwrite(args.outfile, img)
    else:
        temp = tempfile.mktemp(prefix="ckiss-", suffix=".jpg")
        imread.imwrite(temp, img)


if __name__ == "__main__":
    main()
