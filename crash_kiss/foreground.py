"""Functions for determining which parts of an image are in the foreground
and which parts are in the background.

The goal is to use a `threshold` to compare pixels to a `bg_value`
(a background value, usually something to represent the color white).
If we have a pixel P of three dimensions (R, G, B), P is considered to be
a part of the 'foreground' if ANY of R, G, or B is different enough than
`bg_value`. In other words,
`is_foreground = any(color - threshold > threshold for color in pixel)"""

import cv2
import numpy as np

from nphusl import to_husl

from collections import namedtuple
from crash_kiss.config import BLACK, WHITE, FULL_DEPTH
from crash_kiss import util


def find_foreground(img, params):
    """Find the foreground of the image by subracting each RGB element
    in the image by the background. If the background has been reduced
    to a simple int or float, we'll try to avoid calling `np.abs`
    by checking to see if the background value is near 0 or 255."""
    view, bounds = get_foreground_area(img, params.max_depth)
    fg = compare_background(view, params.bg_value, params.threshold)
    util.save_img("fg.jpg", fg * 255)
    return fg, bounds


def trim_foreground(img, foreground, params):
    bounds = get_fg_bounds(img.shape[1], params.max_depth)
    difference = (img.shape[1] - foreground.shape[1]) / 2
    start = int(round(bounds.start - difference))
    stop = int(round(bounds.stop - difference))
    return foreground[:, start: stop], bounds


def compare_background(img, background, threshold):
    assert isinstance(background, int)
    r = background & 0xFF
    g = (background & 0xFF00) >> 8
    b = (background & 0xFF0000) >> 16
    background = r, g, b

    # Reject most pixels based on `threshold` lightness value and some pixels
    # greather than `threshold` based on a combination of H, S, and L
    bg_hue, bg_sat, bg_light = to_husl(background)
    hsl = to_husl(img)  # a 3D array of hue, saturation, and lightness values
    hue, saturation, lightness = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    if (bg_hue, bg_sat, bg_light) == (0., 0., 0.):
        light_enough = lightness > threshold  # 5 works
    else:
        light_enough = np.abs(lightness - bg_light) > threshold

    foreground = light_enough.astype(np.uint8)

    whack_hue = (hue >= 300) | (hue < 22)  # assumes a certain nature of bg artifacts
    too_dim = lightness < 25  # 18 works; assumes subject is always bright
    desaturated = saturation < 40
    foreground = np.logical_and(~(whack_hue & too_dim & desaturated), light_enough).astype(np.uint8)

    foreground = foreground * 255
    foreground = cv2.erode(foreground, (3, 3))
    foreground = (cv2.GaussianBlur(foreground, (3, 3), 0) > 125).astype(np.uint8)
    foreground = fill_holes(foreground.astype(np.uint8))
    foreground = cv2.erode(foreground, (3, 3), iterations=2)

    # Finally, if we have a colorful background check for a significant hue
    # difference. The same threshold is used for lightness and hue given
    # a saturated background color.
    if bg_sat > 10:
        # it's a colorful background
        diff_a = np.abs(hue - bg_hue).astype(np.int)
        diff_b = 360 - np.abs(hue - bg_hue).astype(np.int)
        hue_difference = np.minimum.reduce([diff_a, diff_b])
        colorful_enough = hue_difference >= threshold
    else:
        colorful_enough = np.ones_like(light_enough)

    return foreground.astype(np.bool) & colorful_enough


def get_edges(img: "np.ndarray") -> "np.ndarray":
    # TODO: Finish this experiment
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    util.save_img("gray.jpg", gray)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 10, 50)
    util.save_img("edges.jpg", edges)
    return edges


def fill_holes(foreground: "np.ndarray") -> "np.ndarray":
    holes = foreground * 255
    dkernel = (3, 3)
    holes = holes

    # Fill borders with zeros to assist flood filling
    holes[0, :] = 0
    holes[-1, :] = 0
    holes[:, 0] = 0
    holes[:, -1] = 0

    seed_point = holes.shape[1] // 2, holes.shape[0] // 2
    if foreground[holes.shape[0] // 2, holes.shape[1] // 2]:
        seed_point = (0, 0)
    cv2.floodFill(holes, None, seed_point, 255)

    holes = cv2.bitwise_not(holes)
    fg = foreground.astype(bool) | holes.astype(bool)
    return fg.astype(np.uint8)


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

