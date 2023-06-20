"""General image processing processing functions"""

import os
import imageio
import numpy as np

from crash_kiss import config


def read_img(file_name: str):
    return imageio.imread(file_name)


def save_img(file_name, img):
    opts = {"quality": 100}  # max JPEG quality
    imageio.imwrite(file_name, img, opts=opts)


def combine_images(imgs, horizontal=True):
    axis = 1 if horizontal else 0
    combined = imgs[0]
    for img in imgs[1:]:
        combined = np.append(combined, img, axis=axis)
    return combined


def orient_right_to_left(img):
    return invert_horizontal(img)


def orient_left_to_right(img):
    return img


def orient_down_to_up(img):
    return rotate_cw(img)


def orient_up_to_down(img):
    return rotate_ccw(img)


def invert_horizontal(img):
    return img[::, ::-1]


def invert_vertical(img):
    return img[::-1]


def rotate_180(img):
    return img[::-1, ::-1]


def rotate_ccw(img):
    return img.swapaxes(0, 1)


def rotate_cw(img):
    return rotate_180(img).swapaxes(0, 1)


# Functions for creating numpy slices of images based on a variety of
# RGB index tuples. Covers some of the RGBA (TIFF files) slices that a user
# might want, but not all of them. All permutations of RGB slices are covered.
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


def get_rgb_view(img, rgb_indices):
    """Select a "view" of an image based on RGB indices.
    The views are created with normal `numpy` array slicing,
    they are true mutable views of the original data.

    >>> red_pixels = get_rgb_view(img, [0])
    >>> red_and_blue = get_rgb_view(img, [0, 2])
    >>> rgb = get_rgb_view(img, [0, 1, 2])
    """
    select = _get_rgb_select(img, rgb_indices)
    # We CANNOT use advanced indexing here!
    # Copies of large images are just too expensive.
    if select == tuple(range(img.shape[2])):
        return img  # we've selected ALL of RGB
    elif len(select) == 1:
        return img[:, :, select[0]]  # just a 2-D view
    else:
        try:
            return _rgb_select[select](img)  # a fancy sliced view
        except KeyError:
            from warnings import warn
            warn("RGB select {0} results in a copy!".format(select))
            return view[:, :, select]  # a nasty copy is created


def _get_rgb_select(img, rgb_indices):
    rgb_indices = rgb_indices or range(img.shape[2])
    return tuple(sorted(set(rgb_indices)))


PURPLE = [118, 0, 118]
TEAL = [60, 120, 160]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
PINK = [255, 0, 255]


def reveal_foreground(img, foreground, bounds):
    """Paints the foreground of the foreground selection area
    1) purple if the pixel is in an OUTER quadrant
    or 2) black if the pixel is in an INNER quadrant"""
    start, stop, fg_mid, max_depth = bounds
    foreground = foreground.astype(bool)
    critical_fg = foreground[:, max_depth: -max_depth]
    img[:, start: stop][foreground] = PURPLE
    img[:, start + max_depth: stop - max_depth][critical_fg] = config.BLACK


def reveal_quadrants(img, bounds):
    """Places vertical lines to represent the "quadrants" that
    crash_kiss uses to determine the "smashable area" of the image"""
    start, stop, fg_mid, max_depth = bounds
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


# green, red, yellow, cyan
_EDGE_REVEAL = [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 255, 255]


def reveal_outer_edges(subject, width):
    """Highlights the left, right, upper, and lower edges of an image
    with green, red, yellow, and cyan."""
    if not width:
        width = max(2, subject.img.shape[0] // 50)
    _ = subject.edges  # process edges before mutating the image
    for side, color in zip(subject, _EDGE_REVEAL):
        view = side.view
        left_col = view[::, 0].copy()
        cols = side.edge.copy()
        rows = np.arange(view.shape[0])
        subtracts = [0] + ([1] * (width - 1))
        for n in subtracts:
            nz_cols = cols != 0
            cols[nz_cols] -= n
            view[rows, cols] = color
        view[::, 0] = left_col  # restore edge of image


def get_filename_hints(target, working_dir, out_suffix):
    """Based on the target filename, returns a tuple of

    1) the output directory
    2) the output filename
    3) the chosen output suffix (if `out_suffix` is None)
    4) the output file extension (i.e. '.jpg')

    The user constructs the output file path like this:
    `os.path.join(out_dir, "{0}_{1}.{2}".format(name, suffix, ext)`"""
    suffix = out_suffix or config.OUTPUT_SUFFIX
    out_path = os.path.split(target)
    out_name = out_path[-1]
    out_dir = working_dir or os.path.join(*out_path[:-1])
    out_ext = out_name.split(".")[-1]
    out_name = "".join(out_name.split(".")[:-1])
    return out_dir, out_name, suffix, out_ext

