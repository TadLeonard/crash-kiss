"""threshold
Functions for finding the edges of a subject in an
image-as-a-numpy-array
"""
from __future__ import division
from collections import namedtuple
from itertools import tee
import numpy as np


def config(**kw_overrides):
    conf = dict(_config_defaults)
    conf.update(kw_overrides)
    return conf


_config_defaults = dict(
    neg_sample_size=5,
    threshold=7,
    bg_change_tolerance=3,
)


def _sliding_window(iterable, size):
    it = iter(iterable)
    win = deque((next(it, None) for _ in xrange(size)), maxlen=size)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win


def get_edges(img, config=config()):
    overall, (left, right, up, down) = (
        _get_edge_background(img, config["sample_size"]) 
    left_edge = _get_edge_indices(img, left, config)
    right_edge = _get_edge_indices(img[::, ::-1], left, config)
    return left_edge, right_edge


def _get_edge_background(img, sample_size):
    """Finds overall RGB background value and 
    arrays of average "edge" RGB value for each side of the image
    """
    # (potentially) negative space on all four sides of the image
    left = np.median(img[::, :sample_size:], axis=1)
    right = np.median(img[::, -sample_size::], axis=1)
    up = np.median(img[:sample_size:], axis=0)
    down = np.median(img[-sample_size::], axis=0)

    # find the median RGB value for each side
    medians = left, right, up, down
    medians = [np.median(med, axis=0) for med in medians]

    # sort the four medians, take the mean of the middle two
    medians = np.sort(medians, axis=0)
    overall_neg_space = np.mean(medians[1:3], axis=0)
    return overall_neg_space, (left, right, up, down)


def _get_edge_indices(img, background, config):
    """Finds the 'edge' of the subject of an image based on a background
    value or an array of background values. Returns an array of indices that
    represent the first non-background pixel moving from left to right in 
    the image. Since it finds edges from left to right, it's up to the
    caller to pass in an appropriately inverted or rotated view of the image
    to account for this.
    """
    threshold = config["threshold"]
    bg_change_tolerance = config["bg_change_tolerance"]

    # Here, we see if the background's RGB elements are similar.
    # This way we can do a simple 
    # array - int operation instead of the more expensive array - [R, G, B]
    # or the even pricier array - <array of shape (NROWS, 1, 3)> operation
    while isinstance(background, np.array):
        bmax, bmin = background.max(axis=0), background.min(axis=0)
        diff = (bmax - bmin) < bg_change_tolerance
        if len(background.shape) >= 2:
            diff = np.all(diff)
        if diff:
            background = np.median(background, axis=0)
        else:
            break  # we've simplified the background as much as we can

    foreground = np.all(np.abs(img - background) > threshold, axis=2)
    return np.argmax(foreground, axis=1) 

