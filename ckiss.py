#!/usr/bin/env python
"""
Smashes the things on the left and right side of an image towards the center
"""

import argparse
import os
import glob
import multiprocessing
import pprint
import time
from crash_kiss import edge, config, util
import imread


_conf = config.config()   # default conf values
parser = argparse.ArgumentParser(
	description="Crash two faces into each other",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("target", nargs="?")
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
                         "in --auto-run mode or in normal mode when no "
                         "output file is specified")
parser.add_argument("-W", "--search-suffix",
                    help="specify suffix to search for in working dir "
                         "in --auto-run mode (default is .jpg)")
parser.add_argument("-u", "--output-suffix",
                    help="specify the file name suffix for produced images "
                         "in --auto-run mode or in normal mode when no "
                         "output file is specified")
parser.add_argument("--sequence", type=int, default=0,
                    help="create a sequence of crash kisses from 0 to "
                         "--max-depth in steps of SEQUENCE size")
parser.add_argument("--in-parallel", type=int,
                    default=multiprocessing.cpu_count(),
                    help="generate a sequence of smashed image "
                         "in parallel across N processes")


DEFAULT_OUTPUT_SUFFIX = "smashed"
DEFAULT_INPUT_SUFFIX = ".jpg"
DEFAULT_LATEST = "LAST_CRASH.jpg"


def main():
    args = parser.parse_args()
    if args.auto_run:
        auto_run(args)
    elif args.sequence:
        if args.in_parallel == 1:
            make_sequence(args)
        else:
            make_sequence_parallel(args)
    else:
        run_once(args)
        

def auto_run(args):
    input_suffix = args.search_suffix or DEFAULT_INPUT_SUFFIX
    input_dir = args.working_dir or os.getcwd()
    output_suffix = args.output_suffix or DEFAULT_OUTPUT_SUFFIX
    input_files = gen_new_files(input_dir, input_suffix)
    try:
        _auto_run_loop(output_suffix, input_files, args)
    except KeyboardInterrupt:
        print("User quit with SIGINT")


def _auto_run_loop(suffix, input_files, args):
    for input_file in input_files:
        input_name, input_ext = input_file.split(".")
        loc, name, suffix, ext = _get_filename_hints(
            input_file, args.working_dir, args.output_suffix)
        output_file = "{0}_{1}.{2}".format(name, suffix, ext)
        output_file = os.path.join(loc, output_file)
        run(input_file, output_file, args, save_latest=True)


def gen_new_files(search_dir, search_suffix):
    search_str = "*{0}".format(search_suffix)
    search_dir = os.path.join(search_dir, search_str)
    print("Polling for new files in {0}".format(search_dir))
    old_files = set(glob.glob(search_dir))
    print("Initial files ignored: {0}".format(list(old_files)))
    while True:
        time.sleep(0.1)
        current_files = set(glob.glob(search_dir))
        new_files = current_files - old_files
        for new_file in new_files:
            print("Found new file: {0}".format(new_file))
            yield new_file
        old_files = set(glob.glob(search_dir))
         

def make_sequence(args):
    target = args.target
    max_depth = args.max_depth
    stepsize = args.sequence
    img = imread.imread(target)
    view = util.get_rgb_view(img, args.rgb_select)
    loc, name, suffix, ext = _get_filename_hints(
        args.target, args.working_dir, args.output_suffix)
    template = os.path.join(loc, "{0}_{1}_{2:04d}.{3}")
    params = edge.SmashParams(
        args.max_depth, args.threshold, args.bg_value, args.rgb_select)
   
    image_steps = edge.iter_smash(img, params, stepsize)
    for img, step in image_steps:
        new_file = template.format(name, suffix, step, ext)
        util.save_img(new_file, img)


def make_sequence_parallel(args):
    target = args.target
    max_depth = args.max_depth
    stepsize = args.sequence
    loc, name, suffix, ext = _get_filename_hints(
        args.target, args.working_dir, args.output_suffix)
    tail = "{0}_{1}_{2}.{3}".format(name, suffix, "{0:04d}", ext)
    template = os.path.join(loc, tail)
    params = edge.SmashParams(
        args.max_depth, args.threshold, args.bg_value, args.rgb_select)
    depths = range(max_depth, -stepsize, -stepsize)
    n_procs = args.in_parallel
    pool = multiprocessing.Pool(n_procs)
    start = time.time()
    depth_chunks = list(_chunks(depths, n_procs))
    task_chunks = [(target, params, template, d_chunk)
                   for d_chunk in depth_chunks]
    pool.map(run_parallel_smash, task_chunks)
    print("Smashed {0} images in {1:0.1f} seconds".format(
          len(depths), time.time() - start))


def run_parallel_smash(args):
    edge.parallel_smash(*args)


def _chunks(things, n_chunks):
    n_things = len(things)
    chunksize = max(n_things // n_chunks, 1)
    stop = 0
    remainder = n_things % n_chunks
    while stop < n_things:
        start = stop
        stop += chunksize
        if (n_things - stop) < n_chunks:
            stop = n_things
        chunk = things[start: stop]
        yield things[start: stop]


def run_once(args):
    if args.outfile:
        out_file = args.outfile
    else:
        loc, name, suffix, ext = _get_filename_hints(
            args.target, args.working_dir, args.output_suffix)
        out_file = "{0}_{1}.{2}".format(name, suffix, ext)
        out_file = os.path.join(loc, out_file)
    run(args.target, out_file, args)


def _get_filename_hints(target, working_dir, out_suffix):
    suffix = out_suffix or DEFAULT_OUTPUT_SUFFIX
    out_path = os.path.split(target)
    out_name = out_path[-1]
    out_dir = working_dir or os.path.join(*out_path[:-1])
    out_ext = out_name.split(".")[-1]              
    out_name = "".join(out_name.split(".")[:-1])
    return out_dir, out_name, suffix, out_ext
   

def run(target_file, output_file, args, save_latest=False):
    img = imread.imread(target_file)
    process_img(img, args)
    util.save_img(output_file, img)
    if save_latest:
        util.save_img(DEFAULT_LATEST, img)


def process_img(img, args):
    params = edge.SmashParams(
        args.max_depth, args.threshold, args.bg_value, args.rgb_select) 
    fg, bounds = edge.find_foreground(img, params)
     
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


if __name__ == "__main__":
    main()

