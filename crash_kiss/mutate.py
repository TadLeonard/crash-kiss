"""Functions for mutating a numpy ndarray of an image"""

from __future__ import division
from six.moves import zip
import random
import numpy as np
from crash_kiss.config import WHITE, BLACK
from crash_kiss import util


@profile
def center_smash(img, foreground, maxlen):
    """Move the rows of each subject together until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""
    fg = foreground[:]
    fg_l, fg_r = bisect_img(fg)
    fg_l = util.invert_horizontal(fg_l)
    mid_idx = fg_l.shape[1]  # start idx for LHS of `right`
    l_start = np.argmax(fg_l, axis=1)
    r_start = np.argmax(fg_r, axis=1)
    offs = r_start - l_start
    for img_row, offs, fg_row in zip(img, offs, fg):
        sub_row = img_row[fg_row]
        sub_len = sub_row.shape[0]
        halflen = sub_len // 2  # always truncate; we can't do any better
        start_idx = mid_idx - halflen
        stop_idx = start_idx + sub_len
        img_row[start_idx: stop_idx] = sub_row
        img_row[:start_idx] = WHITE
        img_row[stop_idx:] = WHITE
    img[:maxlen] = WHITE
    img[-maxlen:] = WHITE


def reveal_foreground(subject):
    subject.img[subject.foreground] = BLACK


def reveal_background(subject):
    subject.img[subject.background] = WHITE


def combine_images(imgs, horizontal=True):
    axis = 1 if horizontal else 0
    combined = imgs[0]
    for img in imgs[1:]:
        combined = np.append(combined, img, axis=axis)
    return combined


def bisect_img(img):
    width = img.shape[1]
    half = width // 2
    return img[:, half:], img[:, :half]
 
