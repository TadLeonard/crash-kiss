"""Functions for finding the foreground in an image with a clean
background (i.e. mostly white or black)"""

from __future__ import division
import numpy as np
from crash_kiss.config import BLACK, WHITE, config


def find_foreground(img, background, config):
    """Find the foreground of the image by subracting each RGB element
    in the image by the background. If the background has been reduced
    to a simple int or float, we'll try to avoid calling `np.abs`
    by checking to see if the background value is near 0 or 255."""
    threshold = config["threshold"]
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


class Subject(object):
    """Holds foreground/background data of an image"""

    def __init__(self, img, config=config()):
        self.img = img
        self._foreground_select = None
        self._background_select = None
        self._foreground = None
        self._background = None
        self._config = config

    def select_foreground(self):
        if self._foreground_select is None:
            self._foreground_select = self.img[self.foreground]
        return self._foreground_select

    def select_background(self):
        if self._background_select is None:
            self._background_select = self.img[self.background]
        return self._background_select

    @property
    def foreground(self):
        if self._foreground is None:
            bg = self._config["bg_value"]
            fg = find_foreground(self.img, bg, self._config)
            self._foreground = fg
        return self._foreground 

    @property
    def background(self):
        if self._background is None:
            self._background = ~ self.foreground
        return self._background

