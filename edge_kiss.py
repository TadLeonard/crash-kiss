#!/usr/bin/env python
"""
Command line entry point to `crash kiss` face smashing functions.
"""

from __future__ import print_function
import argparse
import os
import tempfile
import imread
from crash_kiss import outer_edge, util, config


parser = argparse.ArgumentParser(
    description="Crash images into things",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("targets", nargs="+")
parser.add_argument("-s", "--smash", choices=("center", "side"))
parser.add_argument("-d", "--direction", default="lr",
                    help="'ud' up-to-down and so on",
                    choices=("lr", "rl", "ud", "du"))
parser.add_argument("--vertical", action="store_true", default=True)
parser.add_argument("-o", "--outfile", default=None)
parser.add_argument("-O", "--show-with", help="view result with this command",
                    default="display {0} > /dev/null 2>&1")
procargs = parser.add_argument_group("processing")
_conf = config.config()  # defaults
procargs.add_argument("-c", "--bg-change-tolerance", type=int,
                      help="used to reduce the foreground vs. background"
                           "comparison in edge detection",
                      default=_conf["bg_change_tolerance"])
procargs.add_argument("-t", "--threshold",
                      help="min difference between background and foreground "
                           "to determine an edge",
                      default=_conf["threshold"], type=int)
procargs.add_argument("-b", "--bg-value", type=lambda x: map(int, x.split(",")),
                      help="A number to represent the color of the background "
                           "should the user want to manually set it. Use "
                           "'auto' to automatically gather per-row "
                           "background values.",
                      default=_conf["bg_value"])
procargs.add_argument("-B", "--bg-side", choices=outer_edge.side_names,
                      help="Sample the background from this side only. "
                           "Useful when the subject bleeds into one side.")
procargs.add_argument("--bg-sample-size", type=int,
                      help="num pixels of edges space to use in the "
                           "sampling of the background color",
                      default=_conf["bg_sample_size"])
procargs.add_argument("-g", "--rgb-select", default=_conf["rgb_select"],
                      type=lambda x: sorted(map(int, x.split(","))),
                      help="Find edges based on a subset of RGB(A?) by "
                           "passing a comma-sep list of indices")
procargs.add_argument("-r", "--relative-sides", type=lambda x: x.split(","),
                      help="side of the subject (after rotation) to act on",
                      default=_conf["relative_sides"])
procargs.add_argument("--chunksize", type=int, default=_conf["chunksize"],
                      help="Num columns of image to process at once")
debug = parser.add_argument_group("debugging")
debug.add_argument("-e", "--reveal-edges", action="store_true")
debug.add_argument("-E", "--reveal-width", type=int, default=None)


_orientors = dict(
    lr=util.orient_left_to_right, rl=util.orient_right_to_left,
    ud=util.orient_up_to_down, du=util.orient_down_to_up
)


def run():
    # Parse args, combine images passed to ckiss into a single image
    args = parser.parse_args()
    imgs = map(imread.imread, args.targets)
    if not imgs:
        parser.error("Must pass in one or more paths to images")
    elif len(imgs) == 1:
        img = imgs[0]
    elif len(imgs) > 1:
        img = util.combine_images(imgs, horizontal=True)

    # change the orientation of the image if a non-left direction is specified
    working_img = _orientors[args.direction](img)

    # This is where we should process each image for edge detection...
    conf = config.config(threshold=args.threshold,
                         bg_change_tolerance=args.bg_change_tolerance,
                         bg_sample_size=args.bg_sample_size,
                         relative_sides=args.relative_sides,
                         chunksize=args.chunksize,
                         bg_value=args.bg_value,
                         rgb_select=args.rgb_select)
    subject = outer_edge.Subject(img=working_img, config=conf)

    if args.bg_side:
        bg_sample = getattr(subject, args.bg_side).background
        for side in subject:
            side.background = bg_sample 

    if args.reveal_edges:
        _ = subject.edges

    # After this point we're always working with one big, combined image
    if args.smash:
        if args.smash == "center":
            raise NotImplementedError("Can't center smash yet")
        elif args.smash == "side":
            outer_edge.side_smash(subject)

    # Various things to do with the result of our image mutations
    if args.reveal_edges:
        util.reveal_outer_edges(subject, args.reveal_width)
    if args.outfile:
        imread.imwrite(args.outfile, img)
    else:
        temp = tempfile.mktemp(prefix="ckiss-", suffix=".jpg")
        imread.imsave(temp, img)
        os.system(args.show_with.format(temp))


if __name__ == "__main__":
    run()

