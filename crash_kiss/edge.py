"""Functions for finding the foreground in an image with a clean
background (i.e. mostly white or black)"""

from __future__ import division
from collections import namedtuple
import numpy as np
from crash_kiss.config import BLACK, WHITE, FULL_DEPTH, config
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


_NO_MID_FG = 0xFFFF


def center_smash(img, fg, bounds):
    """Move the rows of each subject together until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""
    max_depth = fg.shape[1] // 4
    fg_l = fg[:, :bounds.fg_mid]
    fg_l = util.invert_horizontal(fg_l)
    fg_r = fg[:, bounds.fg_mid:]
     
    l_start = np.argmax(fg_l, axis=1)
    l_start[l_start==0][fg_l[:, 0] == 0] = _NO_MID_FG
    r_start = np.argmax(fg_r, axis=1)
    r_start[r_start==0][fg_r[:, 0] == 0] = _NO_MID_FG

    for irow, ls, rs, frow in zip(img, l_start, r_start, fg):
        if ls == _NO_MID_FG or rs == _NO_MID_FG:
            # nothing in the selection zone
            # no contact can be made
            lshift = max_depth
            rshift = max_depth
        elif rs - ls < max_depth * 2:
            print "skip"
            # nothing in the smashing zone
            # no contact can be made 
            pass
        else:
            # there's something to smash
            print "SKIP"
            l_shift = rs - ls
        irow[lshift:bounds.start+lshift] = irow[:bounds.start]
        irow[:lshift] = WHITE
        irow[bounds.stop-rshift:-rshift] = irow[bounds.stop:]
        irow[-rshift:] = WHITE

    
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


def reveal_foreground(img, foreground, bounds):
    img[:, bounds.start: bounds.stop][foreground] = BLACK


def reveal_background(img, foreground, bounds):
    img[:, bounds.start: bounds.stop][foreground] = WHITE

