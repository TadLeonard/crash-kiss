"""Functions for finding the foreground in an image with a clean
background (i.e. mostly white or black)"""

from __future__ import division
from collections import namedtuple
import numpy as np
from crash_kiss.config import BLACK, WHITE, PURPLE, FULL_DEPTH, config
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


@profile
def center_smash(img, fg, bounds):
    """Move the rows of each subject together until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""
    start, stop, fg_mid = bounds
    rlen = img.shape[1] - stop
    max_depth = fg.shape[1] // 4
    center = start + 2*max_depth
    side_len = fg.shape[1] // 2
    fg_l = fg[:, :bounds.fg_mid]
    fg_l = util.invert_horizontal(fg_l)
    fg_r = fg[:, bounds.fg_mid:]
     
    lstart = np.argmax(fg_l, axis=1)
    lstart[fg_l[:, 0] == 1] = _MID_FG
    rstart = np.argmax(fg_r, axis=1)
    rstart[fg_r[:, 0] == 1] = _MID_FG

    @profile
    def mov_to_center(irow, frow):
        lextra = frow[:fg_mid][::-1].argmin()
        rextra = frow[fg_mid:].argmin()    
        lidx = start + max_depth - lextra
        lidx_fg = max_depth - lextra
        lsubj = irow[lidx: center][frow[lidx_fg: fg_mid]].copy()
        ridx = center + max_depth + rextra
        ridx_fg = fg_mid + max_depth + rextra
        rsubj = irow[center: ridx][frow[fg_mid: ridx_fg]].copy()
        llen = len(lsubj)
        rlen = len(rsubj)
        lmov = max_depth + lextra - llen
        rmov = max_depth + rextra - rlen
        irow[lmov: center] = irow[:center - lmov]
        irow[center - llen: center] = lsubj
        irow[center: -rmov] = irow[center + rmov:]
        irow[center: center + rlen] = rsubj

    for irow, ls, rs, frow in zip(img, lstart, rstart, fg):
        lmov = rmov = max_depth
        if rs == _MID_FG and ls == _MID_FG:
            # NOTE: smash behavior not fully defined yet
            # (if the subject overlaps the middle)
            mov_to_center(irow, frow)
        elif not ls and not rs:
            irow[center + rmov: -rmov] = irow[stop:]
            irow[lmov: start+lmov] = irow[:start]
        elif not ls or not rs and (ls + rs > max_depth):
            # no contact can be made
            irow[center: -rmov] = irow[center + rmov:]
            irow[lmov: center] = irow[:start + lmov]
        elif ls and not rs:
            # no contact is possible, but there's a left-over-center overshoot
            irow[center + rmov: -rmov] = irow[stop:]
            overshoot = ls - max_depth
            assert overshoot > 0
            print rs, ls, overshoot
            irow[lmov: center+overshoot] = irow[:center-ls]
        elif rs:
            # no contact is possible
            # a right-over-center overshoot is possible
            irow[:] = [200, 200, 200] 
        elif (ls >= max_depth) and (rs >= max_depth):
            # no contact can be made but
            # we'll select & shift the foreground differently
            irow[lmov: start+lmov*2] = irow[:start+max_depth]
            irow[center: -rmov] = irow[stop-rmov:]
        elif rs + ls <= side_len:
            # contact will be made (this is the "crash" or "smash")
            irow[:] = [100, 0, 0]  # dark red
            continue
            subj = irow[start: stop][frow]
            subjl = len(subj)
            offs = rs - ls
            fstart = start + fg_mid + offs - (subjl // 2)
            lmov = fstart - start
            rmov = stop - (fstart + subjl)
            irow[fstart: fstart + subjl] = subj
            irow[lmov: start+lmov] = irow[:start]
            r1 = irow[stop-rmov:-rmov]
            s2 = rlen - len(r1)
            r2 = irow[stop+s2:]
            r1[:] = r2
        else:
            # contact won't be made, but white space may cover
            # either side of the subject if we're not careful
            irow[:] = [128, 255, 0]  # bright green
            continue
            lsubj = irow[start: start+rs].copy()
            rsubj = irow[stop-side_len+rs:stop].copy()
            irow[center: -rs] = irow[center+rs:]
            irow[lmov: start+lmov] = irow[:start]
            irow[start+lmov:start+lmov+len(lsubj)] = lsubj 
            irow[center+rs-rmov:center+rs-rmov+len(rsubj)] = rsubj
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


def reveal_foreground(img, foreground, bounds):
    max_depth = foreground.shape[1] // 4
    start, stop, fg_mid = bounds
    critical_fg = foreground[:, max_depth: -max_depth]
    img[:, start: stop][foreground] = PURPLE
    img[:, start+max_depth: stop-max_depth][critical_fg] = BLACK
    img[:, start-1:start+1] = PURPLE
    img[:, stop-1:stop+1] = PURPLE
    img[:, start+max_depth*2] = BLACK
    img[:, start+max_depth] = BLACK
    img[:, stop-max_depth] = BLACK


def reveal_background(img, foreground, bounds):
    img[:, bounds.start: bounds.stop][foreground] = WHITE

