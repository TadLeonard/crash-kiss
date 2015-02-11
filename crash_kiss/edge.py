"""Functions for determining which parts of an image are in the foreground
and which parts are in the background.

The goal is to use a `threshold` to compare pixels to a `bg_value`  
(a background value, usually something to represent the color white).
If we have a pixel P of three dimensions (R, G, B), P is considered to be
a part of the 'foreground' if ANY of R, G, or B is different enough than
`bg_value`. In other words, 
`is_foreground = any(color - threshold > threshold for color in pixel)"""

from __future__ import division, print_function
from collections import namedtuple
from itertools import repeat
import os
import sys
import time
from crash_kiss.config import BLACK, WHITE, FULL_DEPTH
import crash_kiss.util as util
import imread
import numpy as np
from six.moves import range


def find_foreground(img, params):
    """Find the foreground of the image by subracting each RGB element
    in the image by the background. If the background has been reduced
    to a simple int or float, we'll try to avoid calling `np.abs`
    by checking to see if the background value is near 0 or 255."""
    view, bounds = get_foreground_area(img, params.max_depth)
    view = util.get_rgb_view(view, params.rgb_select)
    if len(view.shape) == 2:
        return compare_background(view, params.bg_value, params.threshold)
    rgb_views = [view[:, :, idx] for idx in range(view.shape[-1])]
    
    # the foreground is a 2D array where 1==foreground 0==background
    fg = compare_background(rgb_views[0], params.bg_value, params.threshold)
    for rgb_view in rgb_views[1:]:
        bg = fg == 0
        new_data = compare_background(
            rgb_view[bg], params.bg_value, params.threshold)
        fg[bg] = new_data
    return fg, bounds


def compare_background(img, background, threshold):
    """Compare a 2-D or 3-D image array to a background value given a
    certain threshold"""
    is_num = isinstance(background, int)
    if background - BLACK <= 5:
        diff = img - background > threshold
    elif WHITE - background <= 5:
        diff = background - img > threshold
    else:
        diff = np.abs(img - background) > threshold
    if len(diff.shape) == 3:
        diff = np.any(diff, axis=2)  # we're using a 3D array
    return diff
   

def simplify_background(background, config):
    """See if the background's RGB elements are similar.
    If each element in the background is similar enough, we can do a simple
    array - int operation instead of the more expensive array - [R, G, B]
    or the even pricier array - <array of shape (NROWS, 1, 3)> operation."""
    while isinstance(background, np.ndarray):
        bg_change_tolerance = config["bg_change_tolerance"]
        bmax, bmin = background.max(axis=0), background.min(axis=0)
        diff = (bmax - bmin) < bg_change_tolerance
        if len(background.shape) >= 2:
            diff = np.all(diff)
        if diff:
            background = np.median(background, axis=0)
        else:
            break  # we've simplified the background as much as we can
    return background


# _MID_FG is a placeholder value for foreground data that is at the center
# of the image. It's used to distinguish the fg-at-center case from
# the no-fg-at-all case. This is because `np.argmax` returns 0 if the max
# value is at index 0 (whether that max value is 0 or 1 or anything else!).
_MID_FG = 0xFFFF  # placeholder index for foreground data at the center
_params = "max_depth threshold bg_value rgb_select".split()


class SmashParams(object):
    """A **picklable** container of values that can be sent to 
    a `multiprocessing.Process` object. Usually a `namedtuple` or some
    other simple container would be better, but `namedtuple` is not 
    picklable!"""

    def __init__(self, *args, **kwargs):
        given_args = dict(zip(_params, repeat(None)))
        for name, value in zip(_params, args):
            given_args[name] = value
        for name, value in kwargs.items():  
            given_args[name] = value
        self.__dict__.update(given_args)


def iter_smash(img, params, stepsize=1):
    """Yield control to another function for each iteration of a smash.
    Each time the image is yeilded, the smash progresses by `stepsize`"""
    start = time.time()
    fg, bounds = find_foreground(img, params)  # initial bground mask, bounds
    max_depth = params.max_depth
    yield center_smash(img.copy(), fg, bounds), max_depth  # deepest smash

    # We'll create a background mask (i.e. the foreground selection) with
    # the same shape as the image. This lets us calculate the entire 
    # foreground just once and slice it down to size for each iteration
    # of the smash. This saves lots of CPU cycles.
    total_fg = np.zeros(
        shape=img.shape[:2], dtype=bool)  # 2D mask with same dims as img
    total_fg[:, bounds.start: bounds.stop] = fg
    
    print("Processing...")
    depths = range(max_depth - stepsize, 0, -stepsize)
    for depth in depths:
        yield _smash_at_depth(img, total_fg, depth), depth
        print("Depth: {0:04d}\r".format(depth), end="")
        sys.stdout.flush()
    yield img, 0  # shallowest smash (just the original image)
    print("{0} images in {1:0.1f} seconds".format(
          len(depths) + 2, time.time() - start))


def parallel_smash(target, params, template, depths):
    """Given an input filename, `target`, a `SmashParams` instance,
    a `template` for output filenames, and an iterable of `depths`, write
    a smashed version of the target image to the disk for each depth.
    """
    # The images are *written* and not returned or yielded because it's not
    # efficient to pass huge n-dimensional arrays between processes. Each
    # process reads its own copy of the target image from the disk and writes
    # its own smashed output files to the disk. In my tests, the program is
    # mostly limited by disk IO (even on SSDs). While each additional process
    # below cpu_count() improves performance, it's usually not by huge amounts.
    start = time.time()
    img = imread.imread(target)
    fg, bounds = find_foreground(img, params)  # initial bground mask, bounds
    max_depth = depths[0]
    first_img = center_smash(img.copy(), fg, bounds)
    util.save_img(template.format(max_depth), first_img)

    # We'll create a background mask (i.e. the foreground selection) with
    # the same shape as the image. This lets us calculate the entire 
    # foreground just once and slice it down to size for each iteration
    # of the smash. Not having to recalculate the foreground each time
    # saves lots of CPU cycles. 
    total_fg = np.zeros(
        shape=img.shape[:2], dtype=bool)  # 2D mask with same dims as img
    total_fg[:, bounds.start: bounds.stop] = fg
    for depth in depths[1:]:
        if depth == 0:
            smashed = img
        else:
            smashed = _smash_at_depth(img, total_fg, depth)
        util.save_img(template.format(depth), smashed)
        
    print("Worker process smashed {0} images in {1:0.1f} seconds".format(
          len(depths), time.time() - start))


def _smash_at_depth(img, total_fg, depth):
    """Select a subset of the complete background mask (the foreground)
    and smash that subset of pixels by `depth`"""
    fg, bounds = get_foreground_area(total_fg, depth)
    smashed_img = center_smash(img.copy(), fg, bounds)
    return smashed_img


_smash_data = namedtuple("sdata", "start stop fg_mid "
                                  "max_depth fg_l fg_r "
                                  "mid_left center mid_right "
                                  "side_len")
_row_data = namedtuple("row", "irow ls rs frow")


def center_smash(img, fg, bounds):
    """Move the rows of each subject together until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""
    start, stop, fg_mid = bounds
    max_depth = fg.shape[1] // 4
    fg_l = fg_mid - max_depth
    fg_r = fg_mid + max_depth
    mid_left = start + max_depth
    center = start +  2 * max_depth
    mid_right = center + max_depth
    side_len = fg.shape[1] // 2
    smash_data = _smash_data(start, stop, fg_mid, max_depth, fg_l,
                             fg_r, mid_left, center, mid_right,
                             side_len)

    lfg = fg[:, :bounds.fg_mid]
    lfg = util.invert_horizontal(lfg)
    rfg = fg[:, bounds.fg_mid:]
    lstart = np.argmax(lfg, axis=1)
    lstart[lfg[:, 0] == 1] = _MID_FG
    rstart = np.argmax(rfg, axis=1)
    rstart[rfg[:, 0] == 1] = _MID_FG

    for row_data in zip(img, lstart, rstart, fg):
        irow, ls, rs, frow = row_data
        row_data = _row_data(*row_data)
        lmov = rmov = max_depth
        if not ls and not rs:
            mov_empty_fg(smash_data, row_data)
        elif ls and not rs:
            mov_left_overshoot(smash_data, row_data)
        elif rs and not ls:
            mov_right_overshoot(smash_data, row_data)
        elif rs == _MID_FG or ls == _MID_FG:
            lmov, rmov = smash(smash_data, row_data)
        elif (rs < max_depth) and (ls < max_depth):
            lmov, rmov = smash(smash_data, row_data)
        elif rs + ls <= side_len:
            lmov, rmov = smash(smash_data, row_data)
        else:
            mov_near_collision(smash_data, row_data)
        irow[:lmov] = WHITE
        irow[-rmov:] = WHITE
    return img


def mov_empty_fg(smash, row):
    """Smash a row with an empty foreground area"""
    irow, ls, rs = row[:-1]
    depth = smash.max_depth
    center = smash.center
    irow[smash.mid_right: -depth] = irow[smash.stop:]
    irow[depth: smash.mid_left] = irow[:smash.start]


def mov_left_overshoot(smash, row):
    """Smash a row where the left side overshoots the center line"""
    irow = row.irow
    depth = smash.max_depth
    irow[depth: smash.mid_right] = irow[:smash.center] 
    irow[smash.mid_right: -depth] = irow[smash.stop:]  # no RHS FG


def mov_right_overshoot(smash, row):
    """Smash a row where the right side overshoots the center line"""
    irow = row.irow
    depth = smash.max_depth
    irow[depth: smash.mid_left] = irow[:smash.start]  # no LHS FG
    irow[smash.mid_left: -depth] = irow[smash.center:]


def smash(smash, row):
    fg_mid = smash.fg_mid
    center = smash.center
    max_depth = smash.max_depth
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
    irow[center + fg_r_start + rlen: -rmov] = (
        irow[center + fg_r_start + rlen + rmov:])
    return lmov, rmov


def mov_near_collision(smash, row):
    depth = smash.max_depth
    center = smash.center
    irow = row.irow
    ls, rs = row.ls, row.rs
    irow[depth: center - ls + depth] = irow[: center - ls]
    irow[center + rs - depth: -depth] = irow[center + rs:]

    
def get_foreground_area(img, max_depth):
    bounds = get_fg_bounds(img.shape, max_depth)
    return img[:, bounds.start: bounds.stop], bounds


_fg_bounds = namedtuple("fg_bounds", "start stop fg_mid")


def get_fg_bounds(img_width, max_depth):
    """Returns start, stop idx of the 'smashable foreground'
    area in the middle of an image.

    Indexing the image with `img[:, start:stop]` will successfully
    select the foreground columns."""
    half = img_width // 2
    if max_depth >= half or max_depth == FULL_DEPTH:
        max_depth = img_width // 4
    start = (img_width // 2) - (max_depth * 2)
    stop = start + max_depth * 4
    assert stop - start == max_depth * 4
    fg_mid = (stop - start) // 2
    return _fg_bounds(start, stop, fg_mid)


PURPLE = [118, 0, 118]
TEAL = [60, 120, 160]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
PINK = [255, 0, 255]


def reveal_foreground(img, foreground, bounds):
    """Paints the foreground of the foreground selection area
    1) purple if the pixel is in an OUTER quadrant
    or 2) black if the pixel is in an INNER quadrant"""
    start, stop, fg_mid = bounds
    max_depth = (stop - start) // 4
    critical_fg = foreground[:, max_depth: -max_depth]
    img[:, start: stop][foreground] = PURPLE
    img[:, start + max_depth: stop - max_depth][critical_fg] = BLACK


def reveal_quadrants(img, bounds):
    """Places vertical lines to represent the "quadrants" that
    crash_kiss uses to determine the "smashable area" of the image"""
    start, stop, fg_mid = bounds
    max_depth = (stop - start) // 4
    width = 4
    width = img.shape[1] // 2000 + 1
    lmid = start + max_depth
    mid = start + max_depth * 2
    rmid = stop - max_depth
    img[:, start - width: start + width] = GREEN
    img[:, stop - width: stop + width] = GREEN
    img[:, mid - width: mid + width] = YELLOW
    img[:, lmid - width: lmid + width] = PINK
    img[:, rmid - width: rmid + width] = PINK


def reveal_background(img, foreground, bounds):
    """Paints the background of the foreground selection area teal"""
    img[:, bounds.start: bounds.stop][foreground == 0] = TEAL

