"""Functions for determining which parts of an image are in the foreground
and which parts are in the background.

The goal is to use a `threshold` to compare pixels to a `bg_value`  
(a background value, usually something to represent the color white).
If we have a pixel P of three dimensions (R, G, B), P is considered to be
a part of the 'foreground' if ANY of R, G, or B is different enough than
`bg_value`. In other words, 
`is_foreground = any(color - threshold > threshold for color in pixel)"""

from __future__ import division
from collections import namedtuple
from crash_kiss.config import BLACK, WHITE, FULL_DEPTH
from crash_kiss import util
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


def trim_foreground(img, foreground, params):
    bounds = get_fg_bounds(img.shape[1], params.max_depth)
    difference = (img.shape[1] - foreground.shape[1]) / 2
    start = bounds.start - difference
    stop = bounds.stop - difference
    return foreground[:, start: stop], bounds
 

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
    return diff.astype(np.uint8)
   
    
def get_foreground_area(img, max_depth):
    """Make a slice of the middle of the image based on `max_depth`.
    Returns a view of the middle section of `img` and a `namedtuple`
    of `start, stop, fg_mid, max_depth` integers."""
    bounds = get_fg_bounds(img.shape[1], max_depth)
    return img[:, bounds.start: bounds.stop], bounds


_fg_bounds = namedtuple("fg_bounds", "start stop fg_mid max_depth")


def get_fg_bounds(img_width, max_depth):
    """Returns start, stop idx of the 'crashable foreground'
    area in the middle of an image.

    Indexing the image with `img[:, start:stop]` will successfully
    select the foreground columns."""
    if max_depth > img_width // 4 or max_depth == FULL_DEPTH:
        max_depth = img_width // 4
    start = (img_width // 2) - (max_depth * 2)
    stop = start + max_depth * 4
    fg_mid = (stop - start) // 2
    return _fg_bounds(start, stop, fg_mid, max_depth)

