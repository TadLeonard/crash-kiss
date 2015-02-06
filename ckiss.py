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
parser.add_argument("-b", "--bg-value", type=int,
                    help="A number to represent the color of the background "
                         "should the user want to manually set it. Use "
                         "'auto' to automatically gather per-row "
                         "background values.",
                      default=_conf["bg_value"])
parser.add_argument("-o", "--outfile", default=None)
parser.add_argument("-e", "--reveal-foreground", action="store_true")
parser.add_argument("-E", "--reveal-background", action="store_true")
parser.add_argument("-q", "--reveal-quadrants", action="store_true",
                    help="reveal the inner and outer quadrants of the "
                         "'smashable area' with vertical lines")
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
parser.add_argument("-a", "--auto-run", action="store_true",
                    help="automatically process new images that appear in "
                         "the working directory")
parser.add_argument("-w", "--working-dir",
                    help="specify the directory for newly processed images "
                         "in --auto-run mode")
parser.add_argument("-W", "--working-file-suffix",
                    help="specify suffix to search for in working dir "
                         "in --auto-run mode (default is .jpg)")
parser.add_argument("-u", "--output-suffix",
                    help="specify the suffix for processed images "
                         "in --auto-run mode")


DEFAULT_OUTPUT_SUFFIX = "smashed"
DEFAULT_INPUT_SUFFIX = ".jpg"


def main():
    args = parser.parse_args()
    if not args.auto_run and args.output_suffix:
        parser.error("-u doesn't make sense without -a")
    elif not args.auto_run and args.working_dir:
        parser.error("-w doesn't make sense without -a")
    if args.auto_run:
        auto_run(args)
    else:
        run_once(args)
        

def auto_run(args):
    pass


def run_once(args):
    run(args.target, args.outfile, args)


def gen_new_files(directory):
    initial_glob = glob.glob(os.path.join(directory, "*.jpg")


def run(target_file, output_file, args):
    img = imread.imread(target_file)
    process_img(img, args)
    save_img(img, output_file)
    
 

def process_img(img, args):
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
    if args.reveal_quadrants:
        edge.reveal_quadrants(img, bounds)
    return img


def save_img(img, file_name):
    opts = {"quality": 100}  # max JPEG quality
    imread.imwrite(file_name, img, opts=opts)


if __name__ == "__main__":
    main()

