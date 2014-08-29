"""Functions for mutating a numpy ndarray of an image"""

from six.moves import zip
import numpy as np


def center_smash_image(edges, img):
    """The original "crash kiss" method used to smash two people's
    profiles together in a grotesque "kiss". The rule is: move the
    subjects of each row towards each other until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""


def wall_smash_image(edges, img):
    """Mutates a numpy array of an image so that the subject
    is smashed to an edge of the image boarders."""
    for row_data_group, row in _iter_subject_rows(edges, img):
        _shift_row_left_to_right(row_data_group, row)


def _shift_img_left_to_right(left_edge, right_edge, img):
    target_idx = row.shape[0]  # shift to end of img initially
    rowlens = left 
    rowlen = edge.right_idx - edge.left_idx
    sub_data_l = target_idx - rowlen
    sub_data_r = target_idx
    row[sub_data_l: sub_data_r] = row[edge.left_idx: edge.right_idx]
    target_idx = sub_data_l

    # We've shifted the subject(s) over, now we need to fill
    # the rest of the row with negative space
    row[:sub_data_l] = edge.neg_space_l


def _iter_subject_rows(edges, img):
    """Iterate over edges, pixel rows that contain foreground info
    (i.e. not all whitespace)."""
    for row_data_group, row in zip(edges, img):
        if any(r - l for l, r, _, _ in row_data_group):
            yield row_data_group, row


_L_EDGE_REVEAL = [0, 255, 0]
_R_EDGE_REVEAL = [255, 0, 0]


def reveal_edges(edges, img, inplace=False):
    """Highlights the edges of an image with green (left edge)
    and red (right edge)"""
    new_img = img if inplace else img.copy()
    for row, edge_group in zip(new_img, edges):
        for l, r, _, _ in edge_group:
            row[l-1: l+1] = _L_EDGE_REVEAL
            row[r-1: r+1] = _R_EDGE_REVEAL
    return new_img


def combine_images(imgs, horizontal=True):
    axis = 1 if horizontal else 0
    combined = imgs[0]
    for img in imgs[1:]:
        combined = np.append(combined, img, axis=axis)
    return combined


