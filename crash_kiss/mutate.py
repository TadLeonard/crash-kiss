"""Functions for mutating a numpy ndarray of an image"""

from six.moves import zip
import numpy as np


def center_smash(edges, img):
    """Move the rows of each subject together until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""


def side_smash(subject, out=None, target_edge=None):
    """Mutates a numpy array of an image so that the subject is smashed up
    against one of the image's borders. The left (relative to the subject)
    border is used, so the caller must provide a properly flipped or rotated
    view of the image array to smash the subject against the desired
    border."""
    view = subject.left.view
    if out is None:
        out = view
    if target_edge is None:
        target_edge = np.zeros(out.shape[0])
    subject_width = view.shape[1]
    l_edge = subject.left.edge
    bg_side = subject.right or subject.left
    bg = bg_side.background
    n_cols = view.shape[1]
    bg_edge = (n_cols - l_edge) + target_edge
    zipped = zip(out, l_edge, bg_edge, bg, target_edge)
    for row, l_idx, bg_idx, bg, target_idx in zipped:
        if not l_idx:
            continue
        row[target_idx: bg_idx] = row[l_idx: subject_width]
        row[bg_idx: subject_width] = bg
 

# green, red, yellow, cyan
_EDGE_REVEAL = [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 255, 255]


def reveal_edges(subject, width):
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


def combine_images(imgs, horizontal=True):
    axis = 1 if horizontal else 0
    combined = imgs[0]
    for img in imgs[1:]:
        combined = np.append(combined, img, axis=axis)
    return combined

