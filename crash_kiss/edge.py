"""Functions for finding the foreground in an image with a clean
background (i.e. mostly white or black)"""

from __future__ import division
from collections import namedtuple
import numpy as np
from crash_kiss.config import BLACK, WHITE, FULL_DEPTH
import crash_kiss.util as util


def find_foreground(img, background, threshold):
    """Find the foreground of the image by subracting each RGB element
    in the image by the background. If the background has been reduced
    to a simple int or float, we'll try to avoid calling `np.abs`
    by checking to see if the background value is near 0 or 255."""
    if len(img.shape) == 2:
        return _compare_pixels(img, background, threshold)
    rgb_views = [img[:, :, idx] for idx in range(img.shape[-1])]
    
    # the foreground is a 2D array where 1==foreground 0==background
    fg = _compare_pixels(rgb_views[0], background, threshold)
    for view in rgb_views[1:]:
        bg = fg == 0
        new_data = _compare_pixels(view[bg], background, threshold)
        fg[bg] = new_data
    return fg


def _compare_pixels(img, background, threshold):
    """Compare a 2-D or 3-D image array
    to a background value given a certain threshold"""
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


_MID_FG = 0xFFFF


def center_smash(img, fg, bounds):
    """Move the rows of each subject together until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""
    start, stop, fg_mid = bounds
    max_depth = fg.shape[1] // 4
    fg_l = fg_mid - max_depth
    fg_r = fg_mid + max_depth
    rlen = img.shape[1] - stop
    mid_left = start + max_depth
    center = start + 2*max_depth
    mid_right = center + max_depth
    side_len = fg.shape[1] // 2

    lfg = fg[:, :bounds.fg_mid]
    lfg = util.invert_horizontal(lfg)
    rfg = fg[:, bounds.fg_mid:]
    lstart = np.argmax(lfg, axis=1)
    lstart[lfg[:, 0] == 1] = _MID_FG
    rstart = np.argmax(rfg, axis=1)
    rstart[rfg[:, 0] == 1] = _MID_FG

    def mov_empty_fg(irow):
        """Smash a row with an empty foreground area"""
        irow[center + rmov: -rmov] = irow[stop:]
        irow[lmov: start+lmov] = irow[:start]

    def mov_no_collision(irow, frow):
        """Smash a row whose foreground area will not touch"""
        irow[center: -max_depth] = irow[center + max_depth:]
        irow[max_depth: center] = irow[:start + max_depth]

    def mov_left_overshoot(irow, frow, ls):
        """Smash a row where the left side overshoots the center line"""
        irow[mid_right: -max_depth] = irow[stop:]  # no RHS FG
        irow[max_depth: mid_right] = irow[:center] 

    def mov_right_overshoot(irow, frow, rs):
        """Smash a row where the right side overshoots the center line"""
        irow[max_depth: mid_left] = irow[:start]  # no LHS FG
        irow[mid_left: -max_depth] = irow[center:]

    def smash(irow, frow, ls, rs):
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

    def mov_near_collision(irow, frow, ls, rs):
        irow[lmov: center - ls + lmov] = irow[: center - ls]
        irow[center + rs - rmov: -rmov] = irow[center + rs:]

    for irow, ls, rs, frow in zip(img, lstart, rstart, fg):
        lmov = rmov = max_depth
        if not ls and not rs:
            mov_empty_fg(irow)
        elif ls and not rs:
            mov_left_overshoot(irow, frow, ls)
        elif rs and not ls:
            mov_right_overshoot(irow, frow, rs)
        elif rs == _MID_FG or ls == _MID_FG:
            lmov, rmov = smash(irow, frow, rs, ls)
        elif (rs < max_depth) and (ls < max_depth):
            lmov, rmov = smash(irow, frow, ls, rs)
        elif rs + ls <= side_len:
            lmov, rmov = smash(irow, frow, ls, rs)
        else:
            mov_near_collision(irow, frow, ls, rs)
        irow[:lmov] = WHITE
        irow[-rmov:] = WHITE

    
def get_foreground_area(img, max_depth):
    bounds = _get_fg_bounds(img.shape, max_depth)
    return img[:, bounds.start:bounds.stop], bounds


_fg_bounds = namedtuple("fg_bounds", "start stop fg_mid")


def _get_fg_bounds(img_shape, max_depth):
    """Returns start, stop idx of the 'smashable foreground'
    area in the middle of an image.

    Indexing the image with `img[:, start:stop]` will successfully
    select the foreground columns."""
    width = img_shape[1]
    half = width // 2
    if max_depth >= half or max_depth == FULL_DEPTH:
        max_depth = width // 4
    start = (width // 2) - (max_depth * 2)
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

