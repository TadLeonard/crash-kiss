#!/usr/bin/env python
"""
Crash kiss
--

An image processing art project. Given an input image, this program

1) tries to determine what is the foreground and what is the background
2) crashes foreground on the left into foreground on the right
3) optionally highlights the background and/or foreground
4) optionally creates a sequential crash for making .gif files
"""

import argparse
import glob
import multiprocessing
import pprint
import os
import time

from collections import namedtuple
from moviepy.editor import VideoClip
from crash_kiss import foreground, config, util, crash


parser = argparse.ArgumentParser(
	description=__doc__,
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group = parser.add_argument_group("kiss options")
group.add_argument("target", nargs="?", help="path to an image file to process")
group.add_argument("-b", "--bg-value", type=int,
                    help="A number to represent the color of the background "
                         "should the user want to manually set it. Use "
                         "'auto' to automatically gather per-row "
                         "background values.",
                      default=config.BG_VALUE)
group.add_argument("-c", "--crash", action="store_true")
group.add_argument("-e", "--reveal-foreground", action="store_true")
group.add_argument("-E", "--reveal-background", action="store_true")
group.add_argument("-q", "--reveal-quadrants", action="store_true",
                    help="reveal the inner and outer quadrants of the "
                         "'crashable area' with vertical lines")
group.add_argument("-t", "--threshold",
                    help="min difference between background and foreground ",
                    default=config.THRESHOLD, type=int)
group.add_argument("-d", "--max-depth", type=int,
                    default=config.MAX_DEPTH,
                    help="Max number of pixels that the left and right "
                         "subjects will smoosh into each other. Neither face "
                         "will collapse by more than max_depth")
group.add_argument("-r", "--rgb-select", default=config.RGB_SELECT,
                    type=lambda x: sorted(map(int, x.split(","))),
                    help="Find edges based on a subset of RGB(A?) by "
                         "passing a comma-sep list of indices")
group.add_argument("-o", "--outfile", default=None)
group.add_argument("-a", "--auto-run", action="store_true",
                    help="automatically process new images that appear in "
                         "the working directory")
group.add_argument("-w", "--working-dir",
                    help="specify the directory for newly processed images "
                         "in --auto-run mode or in normal mode when no "
                         "output file is specified")
group.add_argument("-W", "--search-suffix",
                    help="specify suffix to search for in working dir "
                         "in --auto-run mode (default is .jpg)")
group.add_argument("-u", "--output-suffix",
                    help="specify the file name suffix for produced images "
                         "in --auto-run mode or in normal mode when no "
                         "output file is specified")
group.add_argument("--sequence", type=int, default=0,
                    help="create a sequence of crash kisses from 0 to "
                         "--max-depth in steps of SEQUENCE size")
group.add_argument("--animate", type=int, default=0,
                    help="create an mp4 animation of crash kisses from 0 to "
                         "--mas-depth in steps of ANIMATE size")
group.add_argument("--fps", type=int, default=24)
group.add_argument("--in-parallel", type=int,
                    default=multiprocessing.cpu_count(),
                    help="generate a sequence of crashed image "
                         "in parallel across N processes")
group.add_argument("--compression", default="veryfast",
                    choices=("ultrafast", "veryfast", "fast",))


_options = namedtuple("options", "reveal_foreground reveal_background "
                                "crash reveal_quadrants")


def main():
    args = parser.parse_args()
    if not args.target and not args.auto_run:
        parser.error("Specify a target image or use -a/--auto-run mode")
    if args.auto_run:
        run_auto(args)
    elif args.sequence:
        run_sequence(args)
    elif args.animate:
        run_animate(args, args.target, args.outfile)
    else:
        run_once(args)


def run_auto(args):
    """Automatic photo booth mode! Monitors a directory for new files
    and processes them automatically until SIGINT."""
    input_suffix = args.search_suffix or config.INPUT_SUFFIX
    input_dir = args.working_dir or os.getcwd()
    output_suffix = args.output_suffix or config.OUTPUT_SUFFIX
    input_files = gen_new_files(input_dir, input_suffix)
    try:
        _auto_run_loop(output_suffix, input_files, args)
    except KeyboardInterrupt:
        print("User quit with SIGINT")


def _auto_run_loop(suffix, input_files, args):
    """Processes new images in an infinite loop"""
    for input_file in input_files:
        input_name, input_ext = input_file.split(".")
        loc, name, suffix, ext = util.get_filename_hints(
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
    """Carry out a sequence crash (optionally across multiple processes)"""
    start = time.time()  # keep track of total duration
    target = args.target
    stepsize = args.sequence
    img = util.read_img(target)
    bounds = foreground.get_fg_bounds(img.shape[1], args.max_depth)
    max_depth = bounds.max_depth
    crash_params = crash.CrashParams(
        max_depth, args.threshold, args.bg_value, args.rgb_select)
    depths = range(max_depth, -stepsize, -stepsize)
    depths = [d for d in depths if d > 0]
    depths.append(0)
    n_procs = max(args.in_parallel, 1)
    counter = multiprocessing.RawValue("i", len(depths))
    depth_chunks = list(_chunks(depths, n_procs))
    working_dir, output_suffix = args.working_dir, args.output_suffix
    basic_args = (target, working_dir, output_suffix, crash_params, counter)
    task_chunks = [basic_args + (d_chunk,) for d_chunk in depth_chunks]
    task_chunks = [crash.SequenceParams(*args) for args in task_chunks]

    if n_procs > 1:
        procs = [multiprocessing.Process(
                    target=crash.sequence_crash, args=(params,))
                 for params in task_chunks]
        list(map(multiprocessing.Process.start, procs))
        list(map(multiprocessing.Process.join, procs))
    else:
        list(map(crash.sequence_crash, task_chunks))

    print("Crashed {0} images in {1:0.1f} seconds".format(
          len(depths), time.time() - start))


def run_animate(args, target, outfile):
    stepsize = args.animate
    img = util.read_img(target)
    bounds = foreground.get_fg_bounds(img.shape[1], args.max_depth)
    max_depth = bounds.max_depth
    crash_params = crash.CrashParams(
        max_depth, args.threshold, args.bg_value, args.rgb_select)
    depths = range(max_depth, -stepsize, -stepsize)
    depths = [d for d in depths if d > 0]
    depths.append(0)
    n_frames = len(depths)
    n_procs = max(args.in_parallel, 1)

    fps = args.fps
    duration = len(depths) / fps
    img = util.read_img(target)
    options = _options(args.reveal_foreground, args.reveal_background,
                       args.crash, args.reveal_quadrants)
    source_img = util.read_img(target)
    fg, bounds = foreground.find_foreground(source_img, crash_params)

    def make_frame(time):
        frame_no = int(round(time * fps))
        if frame_no >= n_frames:
            frame_no = n_frames - 1
        depth = depths[-frame_no]
        img = source_img.copy()
        if depth:
            params = crash.CrashParams(
                depth, args.threshold, args.bg_value, args.rgb_select)
            new_fg, new_bounds = foreground.trim_foreground(img, fg, params)
            new_img = _process_img(img, new_fg, new_bounds, options)
        else:
            new_img = source_img
        return new_img

    animation = VideoClip(make_frame, duration=duration)
    clip = animation.to_ImageClip(t=duration)
    clip.duration = 0.1
    clip.write_videofile(outfile, fps=fps, audio=False)
    animation.write_videofile("__temp_crash.mp4", fps=fps, audio=False,
                              preset=args.compression,
                              threads=args.in_parallel)
    os.rename("__temp_crash.mp4", outfile)


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
        loc, name, suffix, ext = util.get_filename_hints(
            args.target, args.working_dir, args.output_suffix)
        out_file = "{0}_{1}.{2}".format(name, suffix, ext)
        out_file = os.path.join(loc, out_file)
    _process_and_save(args.target, out_file, args)


def _process_and_save(target_file, output_file, args, save_latest=False):
    """Processes an image ad saves the rusult. Optionally saves the result
    twice (once to LAST_CRASH.jpg) for convenience in the photo booth."""
    img = util.read_img(target_file)
    params = crash.CrashParams(
        args.max_depth, args.threshold, args.bg_value, args.rgb_select)
    fg, bounds = foreground.find_foreground(img, params)
    options = _options(args.reveal_foreground, args.reveal_background,
                       args.crash, args.reveal_quadrants)
    new_img = _process_img(img, fg, bounds, options)
    util.save_img(output_file, new_img)
    if save_latest:
        ext = output_file.split(".")[-1]
        util.save_img(config.LAST_CRASH.format(ext), new_img)


def _process_img(img, foreground, bounds, options):
    """Does none or more of several things to `img` based on the given
    `argparse` options."""
    if options.reveal_foreground:
        util.reveal_foreground(img, foreground, bounds)
    if options.reveal_background:
        util.reveal_background(img, foreground, bounds)
    if options.crash:
        crash.center_crash(img, foreground, bounds)
    if options.reveal_quadrants:
        util.reveal_quadrants(img, bounds)
    return img


if __name__ == "__main__":
    main()

