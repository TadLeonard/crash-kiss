"""Functions for finding the edges of the foreground in an
image-as-a-numpy-array"""

from __future__ import division
from collections import namedtuple
from itertools import tee
import numpy as np
from six.moves import range
from crash_kiss import util


def config(**kw_overrides):
    invalid = set(kw_overrides) - set(_config_defaults)
    if invalid:
        raise Exception("Invalid config keys: {0}".format(invalid))
    conf = dict(_config_defaults)
    conf.update(kw_overrides)
    return conf

side_names = "left", "right", "up", "down"

_config_defaults = dict(
    bg_sample_size=5,
    threshold=10,
    bg_change_tolerance=7,
    relative_sides=("left", "right"),
    chunksize=300,
    bg_value=None,
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
                            for side in side_names)
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
            self._edge = get_edge(self.view, bg, self._config)
            get_all_edges(self.view, bg, self._config)
        return self._edge

    @property
    def background(self):
        if self._background is None:
            user_defined_bg = self._config["bg_value"]
            if user_defined_bg is not None:
                bg = np.empty((self.view.shape[0], 3), dtype=np.uint8)
                bg[::] = user_defined_bg
                self._background = bg
            else:
                s_size = self._config["bg_sample_size"] 
                self._background = get_background(self.view, s_size)
        return self._background

    @background.setter
    def background(self, precomputed_val):
        self._background = precomputed_val

    def __iter__(self): 
        return iter(self.edge)


def get_background(img, sample_size):
    """Returns an array of median RGB values for the background of the image
    based on the values along the image's edges."""
    bg = np.median(img[::, :sample_size:], axis=1)
    bg = bg.reshape((bg.shape[0], 1, bg.shape[1]))
    return bg


@profile
def get_edge(img, background, config):
    """Finds the 'edge' of the subject of an image based on a background
    value or an array of background values. Returns an array of indices that
    represent the first non-background pixel moving from left to right in 
    the image. Since it finds edges from left to right, it's up to the
    caller to pass in an appropriately inverted or rotated view of the image
    to account for this."""
    # The expensive things here are
    # 1) subtraction / comparison of the whole ndarray
    # 2) calling np.abs across the whole ndarray
    # 3) calling np.all across the whole ndarray
    # 4) repeatedly creating slice views of the whole ndarray
    # argmax is relatively cheap
    bg = _simplify_background(background, config)
    bg_is_array = isinstance(bg, np.ndarray)
    edge = np.zeros(img.shape[0], dtype=np.uint16)
    chunks = _column_blocks(img, config["chunksize"])
    for img_chunk, prev_idx in chunks:
      #  img_chunk *= 0.9
        for img_slice, start, stop in _row_slices(img_chunk, edge):
            if bg_is_array:
                bg = background[start: stop]
            fg = _find_foreground(img_slice, bg, config)
            sub_edge = np.argmax(fg, axis=1)
            nz_sub_edge = sub_edge != 0
            sub_edge[nz_sub_edge] += prev_idx
            edge[start: stop] = sub_edge
    return edge


def get_many_edges(img, background, config):
    """Like `get_edge`, but multiple 2D arrays are generated. Each
    generated array represents another "edge" in the image. This way we can
    detect all the edges of a subject that overlaps itself."""
    bg = _simplify_background(background, config)
    bg_is_array = isinstance(bg, np.ndarray)
    chunks = _column_blocks(img, config["chunksize"])
    edge = np.zeros(img.shape[0], dtype=np.uint16)
    for img_chunk, prev_idx in chunks:
        for img_slice, start, stop in _row_slices(img_chunk, edge):
            if bg_is_array:
                bg = background[start: stop]
            fg = _find_foreground(img_slice, bg, config)
            sub_edge = np.argmax(fg, axis=1)
            nz_sub_edge = sub_edge != 0
            sub_edge[nz_sub_edge] += prev_idx
            edge[start: stop] = sub_edge
    return edge


@profile
def get_all_edges(img, background, config): 
    bg = _simplify_background(background, config)
    fg = _find_foreground(img, bg, config)
    return list(_all_edges(fg, config))


@profile
def _all_edges(foreground, config):
    width = foreground.shape[1]
    max_depth = 99
    for row in foreground:
        yield list(_row_edges(row, max_depth))


@profile
def _row_edges(row, max_depth=999):
    stop = 0
    for _ in range(max_depth):
        start = np.argmax(row[stop:]) + stop
        if start == stop:
            break
        stop = np.argmin(row[start:]) + start
        if stop == start:
            yield start, row.shape[0]
            break
        else:
            yield start, stop


def _column_blocks(img, chunksize):
    n_cols = img.shape[1]
    chunksize = min(chunksize, n_cols)
    n_chunks = ((n_cols // chunksize) // 2) + 1 
    chunks = [chunksize * n for n in range(1, n_chunks)]
    if not chunks or chunks[-1] < n_cols:
        chunks.append(n_cols - 1)  # n_cols - 1 + 1 == n_cols
    prev_idx = 0
    for idx in chunks:
        yield img[::, prev_idx: idx + 1], prev_idx
        prev_idx = idx


def _row_slices(img, edge):
    z_edge = edge == 0
    stop = 0
    n_rows = img.shape[0]
    while stop != n_rows:
        img_slice, start, stop = _get_contiguous_slice(img, z_edge, stop)
        yield img_slice, start, stop


def _get_contiguous_slice(img, z_edge, offset):
    z_edge = z_edge[offset:]
    start = np.argmax(z_edge)
    if not start and not z_edge[start]:
        raise StopIteration
    stop = np.argmin(z_edge[start:])
    start += offset
    stop += start
    if stop == start:
        if z_edge[-1]:
            stop = img.shape[0]
        else:
            stop += 1
    return img[start: stop], start, stop 


@profile
def _find_foreground(img, background, config):
    """Find the foreground of the image by subracting each RGB element
    in the image by the background. If the background has been reduced
    to a simple int or float, we'll try to avoid calling `np.abs`
    by checking to see if the background value is near 0 or 255."""
    threshold = config["threshold"]
    is_num = isinstance(background, (float, int))
    if is_num and threshold <= 1:
        diff = img != background 
    elif is_num and background < 50:
        diff = img - background > threshold
    elif is_num and background > 200:
        diff = background - img > threshold
    else:
        diff = np.abs(img - background) > threshold
    return np.all(diff, axis=2)


def _simplify_background(background, config):
    """Here, we see if the background's RGB elements are similar.
    This way we can do a simple 
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

