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

side_names = "left", "right", "up", "down"

_config_defaults = dict(
    bg_sample_size=5,
    threshold=10,
    bg_change_tolerance=7,
    relative_sides=("left", "right"),
    chunksize=300,
    bg_value=None,
    rgb_select=None,
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
        self._sides = tuple(self._make_side(side) for side in side_names)
        self._active_sides = tuple(side for side in self._sides
                                   if side.name in used_sides)
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



_rgb_select = {
    (0, 1): lambda view: view[:, :, :2],
    (1, 2): lambda view: view[:, :, 1:3],
    (0, 2): lambda view: view[:, :, ::2],
    (0, 3): lambda view: view[:, :, ::3],
    (2, 3): lambda view: view[:, :, 2:4],
    (1, 3): lambda view: view[:, :, 1::2],
    (2, 3): lambda view: view[:, :, 2:4],
    (0, 1, 2): lambda view: view[:, :, :3],
    (1, 2, 3): lambda view: view[:, :, 1:4],
}


class Side(object):
    """A view of an image's subject from a certain perspective
    (left to right, up to down, etc)"""

    def __init__(self, orientation, img=None, config=config()):
        self.name = orientation
        self._edge = self._view = self._background = self._img = None
        self._rgb_view = None
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
#
    @property
    def rgb_view(self):
        """Provide a view of the image that is potentially restricted to
        a subset of the RGB(A?) values."""
        if self._rgb_view is None:
            self._set_rgb_view()
        return self._rgb_view

    def _set_rgb_view(self):
        select = self._get_rgb_select()
        view = self.view
        # We CANNOT use advanced indexing here!
        # Copies of large images are just too expensive.
        # This means we get this horrible switch statement.
        if select == range(view.shape[2]):
            self._rgb_view = view
        elif len(select) == 1:
            self._rgb_view = view[:, :, select[0]]
        else:
            try:
                self._rgb_view = _rgb_select[select](view)
            except KeyError:
                from warnings import warn
                warn("RGB select {0} results in a copy!".format(select))
                self._rgb_view = view[:, :, select]

    def _get_rgb_select(self):
        select = self._config["rgb_select"]
        select = select or range(self.img.shape[2])
        return tuple(sorted(set(select)))

    @property
    def edge(self):
        if self._edge is None:
            bg = self.background
            self._edge = get_edge(self.rgb_view, bg, self._config)
        return self._edge

    @property
    def background(self):
        if self._background is None:
            user_defined_bg = self._config["bg_value"]
            if user_defined_bg is not None:
                bg = np.empty((self.rgb_view.shape[0], 3), dtype=np.uint8)
                bg[::] = user_defined_bg
                self._background = bg
            else:
                s_size = self._config["bg_sample_size"]
                self._background = get_background(self.rgb_view, s_size)
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
    if len(bg.shape) >= 2:
        # we have to reshape for comparison to the 3D RGBA array view
        bg = bg.reshape((bg.shape[0], 1, bg.shape[1]))
    else:
        bg = bg.reshape((bg.shape[0], 1))
    return bg


_EDGE_PLACEHOLDER = 0xFFFF  # for valid edges at index 0


#@profile
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
        for img_slice, start, stop in _row_slices(img_chunk, edge):
            if bg_is_array:
                bg = background[start: stop]
            fg = _find_foreground(img_slice, bg, config)
            sub_edge = np.argmax(fg, axis=1)
            nz_sub_edge = sub_edge != 0
            sub_edge[nz_sub_edge] += prev_idx
            edge[start: stop] = sub_edge
            if not prev_idx:
                edge[fg[::, 0]]= _EDGE_PLACEHOLDER
    mask = edge == 0
    edge[edge == _EDGE_PLACEHOLDER] = 0
    return np.ma.masked_array(edge, mask=mask, copy=False)


def _column_blocks(img, chunksize):
    n_cols = img.shape[1]
    chunksize = min(chunksize, n_cols)
    n_chunks = (3 * (n_cols // chunksize) // 4) + 1
    stop_indices = [chunksize * n for n in range(1, n_chunks)]
    if not stop_indices or stop_indices[-1] < n_cols:
        stop_indices.append(n_cols)
    start_indices = [0] + [s - 1 for s in stop_indices[:-1]]
    for start, stop in zip(start_indices, stop_indices):
        yield img[::, start: stop], start


def _row_slices(img, edge):
    z_edge = edge == 0
    stop = 0
    n_rows = img.shape[0]
    while stop != n_rows:
        img_slice, start, stop = _get_contiguous_slice(img, z_edge, stop)
        yield img_slice, start, stop


def _get_contiguous_slice(img, z_edge, offset):
    start = np.argmax(z_edge[offset:]) + offset
    if start == offset and not z_edge[start]:
        raise StopIteration
    stop = np.argmin(z_edge[start:]) + start
    if stop == start:
        if z_edge[-1]:
            stop = img.shape[0]
        else:
            stop += 1
    return img[start: stop], start, stop


def _find_foreground(img, background, config):
    """Find the foreground of the image by subracting each RGB element
    in the image by the background. If the background has been reduced
    to a simple int or float, we'll try to avoid calling `np.abs`
    by checking to see if the background value is near 0 or 255."""
    threshold = config["threshold"]
    is_num = isinstance(background, (float, int))
    if is_num and background < 50:
        diff = img - background > threshold
    elif is_num and background > 200:
        diff = background - img > threshold
    else:
        diff = np.abs(img - background) > threshold
    if len(diff.shape) == 3:
        return np.all(diff, axis=2)  # we're using a 3D array
    else:
        return diff  # it's a 2D array of just one of RGB or A


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

