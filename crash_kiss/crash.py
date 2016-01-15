"""Functions for smooshing images. Given a background mask (a foreground
selection), these functions move pixels in an image around to create
various "crash" effects. Images can have their foregrounds crashed into
the center of the image, the side of the image, and so on."""

from __future__ import print_function, division
from collections import namedtuple
from itertools import repeat
import os
import sys
import time
import numpy as np
from six.moves import zip, range
from crash_kiss import util, foreground
from crash_kiss.config import WHITE


# _MID_FG is a placeholder value for foreground data that is at the center
# of the image. It's used to distinguish the fg-at-center case from
# the no-fg-at-all case. This is because `np.argmax` returns 0 if the max
# value has index 0 (whether that max value is 0 or 1 or anything else!).
_MID_FG = 0xFFFF  # placeholder index for foreground data at the center
_crash_data = namedtuple(
    "sdata", "start stop fg_mid max_depth fg_l fg_r "
             "mid_left center mid_right side_len")
_row_data = namedtuple("row", "irow ls rs frow")


#@profile
def center_crash(img, fg, bounds):
    """Move the rows of each subject together until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""
    start, stop, fg_mid, max_depth = bounds
    fg_l = fg_mid - max_depth
    fg_r = fg_mid + max_depth
    mid_left = start + max_depth
    center = start +  2 * max_depth
    mid_right = center + max_depth
    side_len = fg.shape[1] // 2
    crash_data = _crash_data(start, stop, fg_mid, max_depth, fg_l,
                             fg_r, mid_left, center, mid_right,
                             side_len)

    lfg = fg[:, :bounds.fg_mid]
    lfg = util.invert_horizontal(lfg)
    rfg = fg[:, bounds.fg_mid:]
    lstart = np.argmax(lfg, axis=1)
    lstart[lfg[:, 0] == 1] = _MID_FG
    rstart = np.argmax(rfg, axis=1)
    rstart[rfg[:, 0] == 1] = _MID_FG
    
    lnil = lstart == 0
    rnil = rstart == 0
    rows_empty = np.logical_and(lnil, rnil)
    rows_left = np.logical_and(~rnil, lnil)
    rows_right = np.logical_and(rnil, ~lnil)
    rows_crash = np.logical_or(lstart == _MID_FG, rstart == _MID_FG)
    rows_close = (rstart + lstart) <= side_len
    rows_closer = np.logical_and(lstart < max_depth, rstart < max_depth)
    rows_other = (rows_empty + rows_left + rows_right +
                  rows_crash + rows_close + rows_closer) == 0

    mov_empty_fg_2(crash_data, rows_empty, img)

    for row_data in zip(img, lstart, rstart, fg):
        irow, ls, rs, frow = row_data
        row_data = _row_data(*row_data)
        lmov = rmov = max_depth
        if not ls and not rs:
            mov_empty_fg(crash_data, row_data)
        elif ls and not rs:
            mov_left_overshoot(crash_data, row_data)
        elif rs and not ls:
            mov_right_overshoot(crash_data, row_data)
        elif rs == _MID_FG or ls == _MID_FG:
            lmov, rmov = mov_crash(crash_data, row_data)
        elif (rs < max_depth) and (ls < max_depth):
            lmov, rmov = mov_crash(crash_data, row_data)
        elif rs + ls <= side_len:
            lmov, rmov = mov_crash(crash_data, row_data)
        else:
            mov_near_collision(crash_data, row_data)
        irow[:lmov] = WHITE
        irow[-rmov:] = WHITE
    return img


def _contiguous_chunks(mask, img):
    """Generates contiguous chunks of an image given a mask"""
    start = stop = None
    for idx, val in enumerate(mask):
        if val:
            stop = idx + 1
            if start is None:
                start = idx
        elif stop is not None:
            yield img[start: stop]
            start = stop = None
    if stop is not None:
        yield img[start: stop]


def mov_empty_fg_2(crash, mask, image):
    for img in _contiguous_chunks(mask, image):
        img[:, crash.mid_right: -crash.max_depth] = img[:, crash.stop:]
        img[:, crash.max_depth: crash.mid_left] = img[:, :crash.start]


def mov_empty_fg(crash, row):
    """Crash a row with an empty foreground area"""
    irow, ls, rs = row[:-1]
    depth = crash.max_depth
    center = crash.center
    irow[crash.mid_right: -depth] = irow[crash.stop:]
    irow[depth: crash.mid_left] = irow[:crash.start]


def mov_left_overshoot(crash, row):
    """Crash a row where the left side overshoots the center line"""
    irow = row.irow
    depth = crash.max_depth
    irow[depth: crash.mid_right] = irow[:crash.center] 
    irow[crash.mid_right: -depth] = irow[crash.stop:]  # no RHS FG


def mov_right_overshoot(crash, row):
    """Crash a row where the right side overshoots the center line"""
    irow = row.irow
    depth = crash.max_depth
    irow[depth: crash.mid_left] = irow[:crash.start]  # no LHS FG
    irow[crash.mid_left: -depth] = irow[crash.center:]


def mov_crash(crash, row):
    """Crash a row towards the center for the case where
    the foreground on either side of the image will make contact"""
    fg_mid = crash.fg_mid
    center = crash.center
    max_depth = crash.max_depth
    ls, rs = row.ls, row.rs
    frow, irow = row.frow, row.irow
    lextra = rextra = 0
    if ls == _MID_FG or rs == _MID_FG:
        if ls != _MID_FG or rs != _MID_FG:
            ls = rs = 0
        if ls == _MID_FG:
            lextra = frow[:fg_mid][::-1].argmin()
        if rs == _MID_FG:
            rextra = frow[fg_mid:].argmin()    
    offs = rs - ls
    dist = rs + ls - 1
    ledge_mov = dist // 2  # TRUNCATION less on left
    redge_mov = dist - ledge_mov
    fg_l_stop = fg_r_start = -ls + ledge_mov - 1
    bg_mask = frow[fg_mid + fg_l_stop - max_depth: fg_mid + fg_r_start + max_depth]
    l_bg_mask = bg_mask[:max_depth]
    r_bg_mask = bg_mask[max_depth:]
    llen = np.count_nonzero(l_bg_mask)
    rlen = np.count_nonzero(r_bg_mask)
    lsquash = len(l_bg_mask) - llen - ledge_mov
    rsquash = len(r_bg_mask) - rlen - redge_mov
    lmov = ledge_mov + lsquash + lextra
    rmov = redge_mov + rsquash + rextra
    subj = irow[center + fg_l_stop - max_depth: center + fg_r_start + max_depth]
    subj = subj[bg_mask]
    irow[center + fg_l_stop - llen: center + fg_r_start + rlen] = subj
    
    irow[lmov: center + fg_l_stop - llen] = (
        irow[:center + fg_l_stop - llen - lmov])
    end = -rmov or irow.shape[0]
    irow[center + fg_r_start + rlen: end] = (
        irow[center + fg_r_start + rlen + rmov:])
    return lmov, rmov


def mov_near_collision(crash, row):
    depth = crash.max_depth
    center = crash.center
    irow = row.irow
    ls, rs = row.ls, row.rs
    irow[depth: center - ls + depth] = irow[: center - ls]
    irow[center + rs - depth: -depth] = irow[center + rs:]


class CrashParams(object):
    """A **picklable** container of values that can be sent to 
    a `multiprocessing.Process` object. Usually a `namedtuple` or some
    other simple container would be better, but `namedtuple` is not 
    picklable!"""
    _params = "max_depth threshold bg_value rgb_select".split()

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

