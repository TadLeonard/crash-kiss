"""Functions for finding the edges of the foreground in an
image-as-a-numpy-array"""

from __future__ import division
from collections import namedtuple
from itertools import tee
import numpy as np
from crash_kiss import util


def config(**kw_overrides):
    invalid = set(kw_overrides) - set(_config_defaults)
    if invalid:
        raise Exception("Invalid config keys: {0}".format(invalid))
    conf = dict(_config_defaults)
    conf.update(kw_overrides)
    return conf

_side_names = "left", "right", "up", "down"

_config_defaults = dict(
    neg_sample_size=5,
    threshold=7,
    bg_change_tolerance=3,
    relative_sides=("left", "right"),
)

_orientors = dict(
    left=util.orient_left_to_right,
    right=util.orient_right_to_left,
    up=util.orient_up_to_down,
    down=util.orient_down_to_up,
)


class Subject(object):
    """Container of `Side` instances, which contain arrays that define the
    edges of the foreground of an image"""

    def __init__(self, img=None, config=config()):
        self._config = config
        self._img = img
        used_sides = config["relative_sides"]
        self._sides = tuple(self._make_side(side)
                            if side in used_sides else None
                            for side in _side_names)
        self._active_sides = filter(None, self._sides)
        self.left, self.right, self.up, self.down = self._sides
        self.img = img

    def _make_side(self, orientation):
        return Side(orientation, img=self.img, config=self._config)

    @property
    def img(self):
        return self._img
    
    @img.setter
    def img(self, img):
        self._img = img
        for side in self:
            side.img = img

    @property
    def background(self):
        return np.array([side.background for side in self])
   
    @background.setter
    def background(self, precomputed_background):
        for side in self:
            side.background = precomputed_background

    @property
    def edges(self):
        return np.array([side.edge for side in self])

    def __iter__(self):
        for side in self._active_sides:
            yield side


class Side(object):
    """A view of an image's subject from a certain perspective
    (left to right, up to down, etc)"""

    def __init__(self, orientation, img=None, config=config()):
        self._edge = self._view = self._background = self._img = None
        self._relative_side = orientation
        self._orient = _orientors[orientation]
        self.img = img
        self._config = config

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, new_img):
        if self._img is not None:
            self._view = None
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
            thresh = self._config["threshold"]
            bg_delta = self._config["bg_change_tolerance"]
            edge = get_edge(self.view, bg, thresh, bg_delta)
            self._edge = edge
        return self._edge

    @property
    def background(self):
        if self._background is None:
            s_size = self._config["neg_sample_size"] 
            self._background = get_background(self.view, s_size)
        return self._background

    @background.setter
    def background(self, precomupted_background):
        self._background = precomputed_background

    def __iter__(self): 
        return iter(self.edge)


def get_background(img, sample_size):
    """Returns an array of median RGB values for the background of the image
    based on the values along the image's edges."""
    bg = np.median(img[::, :sample_size:], axis=1)
    bg = bg.reshape((bg.shape[0], 1, bg.shape[1]))
    return bg


@profile
def get_edge(img, background, threshold, bg_change_tolerance):
    """Finds the 'edge' of the subject of an image based on a background
    value or an array of background values. Returns an array of indices that
    represent the first non-background pixel moving from left to right in 
    the image. Since it finds edges from left to right, it's up to the
    caller to pass in an appropriately inverted or rotated view of the image
    to account for this."""
    # The expensive things here are
    # 1) calling doing a 3D array subtraction
    # 2) calling np.abs across the whole ndarray
    # 3) calling np.all across the whole ndarray
    # argmax is relatively cheap
    background = _reduce_background(background, bg_change_tolerance)
    n_rows, n_cols = img.shape[:2]
    chunksize = n_cols // 10
    chunks = [chunksize * n for n in range(1, 7)]
    chunks.append(n_cols - 1)  # look in small chunks until after halfway
    prev_idx = 0
    # non-edge indices will be masked
    edge = np.ma.array(np.zeros(n_rows), mask=np.ones(n_rows))
    img_view = img.view(np.ma.MaskedArray)
    for idx in chunks:
        img_slice = img_view[::, prev_idx: idx + 1]
        foreground = _find_foreground(img_slice, background, threshold)
        sub_edge = np.ma.argmax(foreground, axis=1).view(np.ma.MaskedArray)
        sub_edge[sub_edge == 0] = np.ma.masked
        sub_edge += prev_idx
        edge[~sub_edge.mask] = sub_edge[~sub_edge.mask]
        img_slice.mask &= sub_edge.mask
        prev_idx = idx
    return edge.view(np.ma.MaskedArray)


@profile
def _find_foreground(img, background, threshold):
    """Find the foreground of the image by subracting each RGB element
    in the image by the background. If the background has been reduced
    to a simple int or float, we'll try to avoid calling `np.abs`
    by checking to see if the background value is near 0 or 255."""
    is_num = isinstance(background, (float, int))
    if is_num and background < 50:
       diff = img - background > threshold
    elif is_num and background > 200:
       diff = background - img > threshold
    else:
       diff = np.ma.abs(img - background) > threshold
    return np.ma.all(diff, axis=2)


def _reduce_background(background, bg_change_tolerance):
    """Here, we see if the background's RGB elements are similar.
    This way we can do a simple 
    array - int operation instead of the more expensive array - [R, G, B]
    or the even pricier array - <array of shape (NROWS, 1, 3)> operation."""
    while isinstance(background, np.ndarray):
        bmax, bmin = background.max(axis=0), background.min(axis=0)
        diff = (bmax - bmin) < bg_change_tolerance
        if len(background.shape) >= 2:
            diff = np.all(diff)
        if diff:
            background = np.median(background, axis=0)
        else:
            break  # we've simplified the background as much as we can
    return background


def _sliding_window(iterable, size):
    it = iter(iterable)
    win = deque((next(it, None) for _ in xrange(size)), maxlen=size)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win

