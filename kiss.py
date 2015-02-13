#!/usr/bin/env python
"""
Crash kiss
--

An image processing art project. Given an input image, this program

1) tries to determine what is the foreground and what is the background
2) smashes foreground on the left into foreground on the right
3) optionally highlights the background and/or foreground
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
	description=__doc__,
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("target", nargs="?", help="path to an image file to process")
parser.add_argument("-b", "--bg-value", type=int,
                    help="A number to represent the color of the background "
                         "should the user want to manually set it. Use "
                         "'auto' to automatically gather per-row "
                         "background values.",
                      default=_conf["bg_value"])
parser.add_argument("-s", "--smash", action="store_true")
parser.add_argument("-e", "--reveal-foreground", action="store_true")
parser.add_argument("-E", "--reveal-background", action="store_true")
parser.add_argument("-q", "--reveal-quadrants", action="store_true",
                    help="reveal the inner and outer quadrants of the "
                         "'smashable area' with vertical lines")
parser.add_argument("-t", "--threshold",
                    help="min difference between background and foreground "
                         "to determine an edge",
                    default=_conf["threshold"], type=int)
parser.add_argument("-d", "--max-depth", type=int,
                    default=_conf["max_depth"],
                    help="Max number of pixels that the left and right "
                         "subjects will smoosh into each other. Neither face "
                         "will collapse by more than max_depth")
parser.add_argument("-r", "--rgb-select", default=_conf["rgb_select"],
                    type=lambda x: sorted(map(int, x.split(","))),
                    help="Find edges based on a subset of RGB(A?) by "
                         "passing a comma-sep list of indices")
parser.add_argument("-o", "--outfile", default=None)
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
        run_auto(args)
    elif args.sequence:
        if args.in_parallel == 1:
            run_sequence(args)
        else:
            run_sequence_parallel(args)
    else:
        run_once(args)
        

def run_auto(args):
    """Automatic photo booth mode! Monitors a directory for new files
    and processes them automatically until SIGINT."""
    input_suffix = args.search_suffix or DEFAULT_INPUT_SUFFIX
    input_dir = args.working_dir or os.getcwd()
    output_suffix = args.output_suffix or DEFAULT_OUTPUT_SUFFIX
    input_files = gen_new_files(input_dir, input_suffix)
    try:
        _auto_run_loop(output_suffix, input_files, args)
    except KeyboardInterrupt:
        print("User quit with SIGINT")


def _auto_run_loop(suffix, input_files, args):
    """Processes new images in an infinite loop"""
    for input_file in input_files:
        input_name, input_ext = input_file.split(".")
        loc, name, suffix, ext = _get_filename_hints(
            input_file, args.working_dir, args.output_suffix)
        output_file = "{0}_{1}.{2}".format(name, suffix, ext)
        output_file = os.path.join(loc, output_file)
        _process_and_save(input_file, output_file, args, save_latest=True)


def _gen_new_files(search_dir, search_pattern):
    """Searches `search_dir` for files matching `search_pattern`.
    Generates the newly discovered file names one at a time."""
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
        old_files = current_files = set(glob.glob(search_dir))
         

def run_sequence(args):
    """Pull together `argparse` args to carry out a sequence smash.
    Writes a series of images for a range of smash depths."""
    target = args.target
    stepsize = args.sequence
    img = imread.imread(target)
    view = util.get_rgb_view(img, args.rgb_select)
    loc, name, suffix, ext = _get_filename_hints(
        args.target, args.working_dir, args.output_suffix)
    template = os.path.join(loc, "{0}_{1}_{2:04d}.{3}")
    bounds = edge.get_fg_bounds(img.shape[1], args.max_depth)
    max_depth = (bounds.stop - bounds.start ) // 4  # actual depth
    params = edge.SmashParams(
        max_depth, args.threshold, args.bg_value, args.rgb_select)
    image_steps = edge.iter_smash(img, params, stepsize)
    for img, step in image_steps:
        new_file = template.format(name, suffix, step, ext)
        util.save_img(new_file, img)


def run_sequence_parallel(args):
    """Carry out a sequence smash across multiple processes"""
    target = args.target
    stepsize = args.sequence
    loc, name, suffix, ext = _get_filename_hints(
        args.target, args.working_dir, args.output_suffix)
    tail = "{0}_{1}_{2}.{3}".format(name, suffix, "{0:04d}", ext)
    template = os.path.join(loc, tail)
    img = imread.imread(target)
    bounds = edge.get_fg_bounds(img.shape[1], args.max_depth)
    max_depth = bounds.max_depth  # actual max depth!
    params = edge.SmashParams(
        max_depth, args.threshold, args.bg_value, args.rgb_select)
    depths = range(max_depth, -stepsize, -stepsize)
    depths = [d for d in depths if d > 0]
    depths.append(0)
    n_procs = args.in_parallel
    pool = multiprocessing.Pool(n_procs)
    start = time.time()
    depth_chunks = list(_chunks(depths, n_procs))
    task_chunks = [(target, params, template, d_chunk)
                   for d_chunk in depth_chunks]
    pool.map(_run_parallel_smash, task_chunks)
    print("Smashed {0} images in {1:0.1f} seconds".format(
          len(depths), time.time() - start))


def _run_parallel_smash(args):
    """Unapacks args for use in call to `pool.map`"""
    # unpacks args for `pool.map` usage
    edge.parallel_smash(*args)


def _chunks(things, n_chunks):
    """Creates `n_chunks` contiguous slices of `things`"""
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
    """Process and save an image just once"""
    if args.outfile:
        out_file = args.outfile
    else:
        loc, name, suffix, ext = _get_filename_hints(
            args.target, args.working_dir, args.output_suffix)
        out_file = "{0}_{1}.{2}".format(name, suffix, ext)
        out_file = os.path.join(loc, out_file)
    _process_and_save(args.target, out_file, args)


def _get_filename_hints(target, working_dir, out_suffix):
    """Based on the target filename, returns a tuple of

    1) the output directory
    2) the output filename
    3) the chosen output suffix (if `out_suffix` is None)
    4) the output file extension (i.e. '.jpg')
    
    The user constructs the output file path like this:
    `os.path.join(out_dir, "{0}_{1}.{2}".format(name, suffix, ext)`"""
    suffix = out_suffix or DEFAULT_OUTPUT_SUFFIX
    out_path = os.path.split(target)
    out_name = out_path[-1]
    out_dir = working_dir or os.path.join(*out_path[:-1])
    out_ext = out_name.split(".")[-1]              
    out_name = "".join(out_name.split(".")[:-1])
    return out_dir, out_name, suffix, out_ext
   

def _process_and_save(target_file, output_file, args, save_latest=False):
    """Processes an image ad saves the rusult. Optionally saves the result
    twice (once to DEFAULT_SMASH.jpg) for convenience in the photo booth."""
    img = imread.imread(target_file)
    new_img = _process_img(img, args)
    util.save_img(output_file, new_img)
    if save_latest:
        ext = output_file.split(".")[-1]
        util.save_img(DEFAULT_LATEST.format(ext), new_img)



def _process_img(img, args):
    """Does none or more of several things to `img` based on the given
    `argparse` args."""
    params = edge.SmashParams(
        args.max_depth, args.threshold, args.bg_value, args.rgb_select) 
    fg, bounds = edge.find_foreground(img, params)
     
    # Various things to do with the result of our image mutations
    if args.reveal_foreground:
        util.reveal_foreground(img, fg, bounds)
    if args.reveal_background:
        util.reveal_background(img, fg, bounds)
    if args.smash:
        crash.center_smash(img, fg, bounds)
    if args.reveal_quadrants:
        util.reveal_quadrants(img, bounds)
    return img


if __name__ == "__main__":
    main()

