"""Functions for finding the edges of the foreground in an
image-as-a-numpy-array"""

from __future__ import division
from collections import namedtuple
from itertools import tee
import numpy as np
from crash_kiss import util


class Edges(object):
    """Container of `Edge` instances, which contain arrays that define the
    edges of the foreground of an image"""

    def __init__(self, img=None, config=config()):
        self._img = img
        self._config = config()
        self.left = self.right = self.up = self.down = None
        self._edges = self.left, self.right, self.up, self.down
        
    def __iter__(self):
        for edge in self._edges:
            if edge is not None:
                yield edge

    @property
    def background(self):
        pass

    @property
    def foreground(self):
        pass

    @property
    def bg_median(self):
        bgs = [side.background for side in self]
        sorted_bgs = np.sort(bgs, axis=0)
        if len(bgs) == 3:
            pass
        elif len(bgs) == 4:
            pass
        else:
            pass


class Edge(object):
    """A single edge of the foreground of an image"""

    def __init__(self, orientation, config=config()):
        self._relative_side = orientation
        self._orient = _orientators[orientation]
        self._view = None
        self._background = None
        self.img = None
        self._config = config
        self.find_background(img)
        self.find_edge(img) 

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, new_img):
        if new_img is None:
            del self.img
        else:
            self._img = new_img

    @property
    def view(self):
        if self._view is None:
            self._view = self._orient(self.img)
        return self._view

    @property
    def edge(self):
        if self._edge is None:
            bg = self.background
            self._edge = get_edge(self.view, bg, config)
        return self._edge

    @property
    def background(self):
        if self._background is None:
            s_size = self._config["neg_sample_size"] 
            self._background = get_background(self.view, s_size)
        return self._background

    def __iter__(self): 
        return iter(self.edge)


def get_background(img, config=config()):
    """Returns an array of median RGB values for the background of the image
    based on the values along the image's edges."""
    sample_size = config["neg_sample_size"]
    return np.median(img[::, :sample_size:], axis=0)


def get_edge(img, config=config()):
    """Finds the 'edge' of the subject of an image based on a background
    value or an array of background values. Returns an array of indices that
    represent the first non-background pixel moving from left to right in 
    the image. Since it finds edges from left to right, it's up to the
    caller to pass in an appropriately inverted or rotated view of the image
    to account for this."""
    threshold = config["threshold"]
    bg_change_tolerance = config["bg_change_tolerance"]

    # Here, we see if the background's RGB elements are similar.
    # This way we can do a simple 
    # array - int operation instead of the more expensive array - [R, G, B]
    # or the even pricier array - <array of shape (NROWS, 1, 3)> operation
    while isinstance(background, type(np.array)):
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


def config(**kw_overrides):
    invalid = set(kw_overrides) - set(_config_defaults)
    if invalid:
        raise Exception("Invalid config keys: {0}".format(invalid))
    conf = dict(_config_defaults)
    conf.update(kw_overrides)
    return conf


_config_defaults = dict(
    neg_sample_size=5,
    threshold=7,
    bg_change_tolerance=3,
    use_relative_sides=("left", "right", "up", "down"),
)


_orientors = dict(
    left=util.orient_left_to_right,
    right=util.orient_right_to_left,
    up=util.orient_up_to_down,
    down=util.orient_down_to_up,
)


def _sliding_window(iterable, size):
    it = iter(iterable)
    win = deque((next(it, None) for _ in xrange(size)), maxlen=size)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win

