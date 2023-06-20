"""Functions for smooshing images. Given a background mask (a foreground
selection), these functions move pixels in an image around to create
various "crash" effects. Images can have their foregrounds crashed into
the center of the image, the side of the image, and so on."""

from collections import namedtuple
from itertools import repeat
import os
import sys
import time
import numpy as np
from crash_kiss import util, foreground
from crash_kiss.config import WHITE
from crash_kiss.omp_smoosh import smoosh


# _MID_FG is a placeholder value for foreground data that is at the center
# of the image. It's used to distinguish the fg-at-center case from
# the no-fg-at-all case. This is because `np.argmax` returns 0 if the max
# value has index 0 (whether that max value is 0 or 1 or anything else!).
_MID_FG = 0xFFFF  # placeholder index for foreground data at the center
_crash_data = namedtuple(
    "sdata", "start stop fg_mid max_depth fg_l fg_r "
             "mid_left center mid_right side_len")
_row_data = namedtuple("row", "irow ls rs frow")


def center_crash(img, fg, bounds, background_value):
    start, stop, fg_mid, depth = bounds
    foreground = np.zeros(img.shape[:2], dtype=np.uint8)
    foreground[:, start: stop] = fg
    smoosh(img, foreground, depth, background_value)
    return img


def _old_center_crash(img, fg, bounds):
    """Move the rows of each subject together until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""
    start, stop, fg_mid, depth = bounds
    fg_l = fg_mid - depth
    fg_r = fg_mid + depth
    mid_left = start + depth
    center = start +  2 * depth
    mid_right = center + depth
    side_len = fg.shape[1] // 2

    lfg = fg[:, :bounds.fg_mid]
    lfg = util.invert_horizontal(lfg)
    rfg = fg[:, bounds.fg_mid:]
    lstart = np.argmax(lfg, axis=1)
    rstart = np.argmax(rfg, axis=1)
    overlap = np.logical_or(rfg[:, 0], lfg[:, 0])
    foreground = np.zeros(img.shape[:2], dtype=np.uint8)
    foreground[:, start: stop] = fg
    lnil = np.logical_and(lstart == 0, ~overlap)
    rnil = np.logical_and(rstart == 0, ~overlap)

    rows_empty = np.logical_and(lnil, rnil)
    for chunk, _ in _contiguous_chunks(rows_empty, img):
        cpy = chunk.copy()
        chunk[:, mid_left: -depth] = chunk[:, center:]
        chunk[:, depth: mid_right] = cpy[:, :center]
        chunk[:, :depth] = WHITE
        chunk[:, -depth:] = WHITE

    # Move rows with subject only on left side OR no subject at all
    rows_left = np.logical_and(~lnil, rnil)
    for chunk, _ in _contiguous_chunks(rows_left, img):
        cpy = chunk.copy()
        chunk[:, mid_left: -depth] = chunk[:, center:]
        chunk[:, depth: mid_right] = cpy[:, :center]
        chunk[:, :depth] = WHITE
        chunk[:, -depth:] = WHITE

    # Move rows with subject only on right side
    rows_right = np.logical_and(~rnil, lnil)
    for chunk, _ in _contiguous_chunks(rows_right, img):
        cpy = chunk.copy()
        chunk[:, depth: mid_right] = chunk[:, :center]
        chunk[:, mid_left: -depth] = cpy[:, center:]
        chunk[:, :depth] = WHITE
        chunk[:, -depth:] = WHITE

    # Move rows with foreground overlapping the center
    chunks = _contiguous_chunks(overlap, img, foreground, lfg, rfg)
    for chunk, (f, _lfg, _rfg) in chunks:
        l = np.argmin(_lfg, axis=1)
        r = np.argmin(_rfg, axis=1)
        left_overlaps = (l > r).astype(np.uint8)
        smoosh.smoosh_overlap(chunk, f, left_overlaps, depth)

    # Move rows with subjects that are close together
    rows_close = np.logical_and(~rnil, ~lnil)
    rows_close[overlap] = 0
    chunks = _contiguous_chunks(
        rows_close, img, lstart, rstart, foreground)
    for chunk, (l, r, f) in chunks:
        smoosh.smoosh(chunk, l, r, f, depth)

    return img


def _contiguous_chunks(mask, img, *masks):
    idx = 0
    while idx <= mask.size - 1:
        start = np.argmax(mask[idx:]) + idx
        if start == idx and not mask[idx]:
            break
        stop = np.argmin(mask[start:]) + start
        if start == mask.size - 1:
            yield img[start: start + 1], (m[start: start + 1] for m in masks)
            break
        elif stop == start:
            if mask[start + 1]:
                yield img[start:], (m[start:] for m in masks)
            else:
                yield img[start: start + 1], (m[start: start + 1] for m in masks)
            break
        else:
            yield img[start: stop], (m[start: stop] for m in masks)
        idx = stop + 1


class CrashParams(object):
    """A **picklable** container of values that can be sent to
    a `multiprocessing.Process` object. Usually a `namedtuple` or some
    other simple container would be better, but `namedtuple` is not
    picklable!"""
    _params = "max_depth threshold bg_value".split()

    def __init__(self, *args, **kwargs):
        given_args = dict(zip(self._params, repeat(None)))
        for name, value in zip(self._params, args):
            given_args[name] = value
        for name, value in kwargs.items():
            given_args[name] = value
        self.__dict__.update(given_args)

    def __iter__(self):
        for param_name in self._params:
            yield self.__dict__[param_name]


class SequenceParams(CrashParams):
    """A picklable record to pass to `parallel_crash` for running a
    crash over multiple processes"""
    _params = ("target working_dir output_suffix crash_params "
               "counter depths".split())


def sequence_crash(params):
    """Given an input filename `target`, a `CrashParams` instance,
    and an iterable of `depths`, write a crashed version of the target
    image to the disk for each depth."""
    # The images are *written* and not returned or yielded because it's not
    # efficient to pass huge n-dimensional arrays between processes. Each
    # process reads its own copy of the target image from the disk and writes
    # its own crashed output files to the disk. In my tests, the program is
    # mostly limited by disk IO (even on SSDs). While each additional process
    # below cpu_count() improves performance, it's usually not by huge amounts.
    start = time.time()  # keep track of duration to show how cool we are

    loc, name, suffix, ext = util.get_filename_hints(
        params.target, params.working_dir, params.output_suffix)
    tail = "{0}_{1}_{2}.{3}".format(name, suffix, "{0:04d}", ext)
    template = os.path.join(loc, tail)
    img = util.read_img(params.target)
    fg, bounds = foreground.find_foreground(img, params.crash_params)
    max_depth = params.depths[0]
    first_img = center_crash(img.copy(), fg, bounds)
    util.save_img(template.format(max_depth), first_img)
    _print_count(params.counter)

    # We'll create a background mask (i.e. the foreground selection) with
    # the same shape as the image. This lets us calculate the entire
    # foreground just once and slice it down to size for each iteration
    # of the crash. Not having to recalculate the foreground each time
    # saves lots of CPU cycles.
    total_fg = np.zeros(shape=img.shape[:2], dtype=bool)  # a 2D mask
    total_fg[:, bounds.start: bounds.stop] = fg
    for depth in params.depths[1:]:
        if depth == 0:
            crashed = img
        else:
            crashed = _crash_at_depth(img, total_fg, depth)
        util.save_img(template.format(depth), crashed)
        _print_count(params.counter)


def _print_count(counter):
    counter.value -= 1
    print("Remaining: {0:04d}\r".format(counter.value), end="")
    sys.stdout.flush()


def _crash_at_depth(img, total_fg, depth):
    """Select a subset of the complete background mask (the foreground)
    and crash that subset of pixels by `depth`"""
    fg, bounds = foreground.get_foreground_area(total_fg, depth)
    crashed_img = center_crash(img.copy(), fg, bounds)
    return crashed_img

