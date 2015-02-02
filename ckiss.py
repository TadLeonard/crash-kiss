#!/usr/bin/env python
"""
Smashes the things on the left and right side of an image towards the center
"""

import argparse
from crash_kiss import edge, config, util
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
parser.add_argument("-t", "--threshold",
                    help="min difference between background and foreground "
                         "to determine an edge",
                    default=_conf["threshold"], type=int)
parser.add_argument("-s", "--smash", action="store_true")
parser.add_argument("-d", "--max-depth", type=int,
                    default=_conf["max_depth"],
                    help="Max number of pixels that the left and right "
                         "subjects will smoosh into each other. Neither face "
                         "will collapse by more than max_depth")
parser.add_argument("-r", "--rgb-select", default=_conf["rgb_select"],
                    type=lambda x: sorted(map(int, x.split(","))),
                    help="Find edges based on a subset of RGB(A?) by "
                         "passing a comma-sep list of indices")


def main():
    args = parser.parse_args()
    img = imread.imread(args.target)
    view, bounds = edge.get_foreground_area(img, args.max_depth)
    view = util.get_rgb_view(view, args.rgb_select)
    fg = edge.find_foreground(view, args.bg_value, args.threshold)
     
    # Various things to do with the result of our image mutations
    if args.reveal_foreground:
        edge.reveal_foreground(img, fg, bounds)
    if args.reveal_background:
        edge.reveal_background(img, fg, bounds)
    if args.smash:
        edge.center_smash(img, fg, bounds)
    opts = {"quality": 100}  # no JPEG compression
    if args.outfile:
        imread.imwrite(args.outfile, img, opts=opts)
    else:
        temp = tempfile.mktemp(prefix="ckiss-", suffix=".jpg")
        imread.imwrite(temp, img, opts=opts)


if __name__ == "__main__":
    main()

