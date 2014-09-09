"""Functions for mutating a numpy ndarray of an image"""

from six.moves import zip
import numpy as np


def center_smash_image(edges, img):
    """The original "crash kiss" method used to smash two people's
    profiles together in a grotesque "kiss". The rule is: move the
    subjects of each row towards each other until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""


def wall_smash_image(subject):
    """Mutates a numpy array of an image so that the subject
    is smashed up against one of the image's borders."""
    im = subject.img
    target = im.shape[1]
    rowlens = subject.right.edge - subject.left.edge
    
    for row, l, r in zip(im, subject.left, subject.right):
        pass
        

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


# green, red, yellow, cyan
_EDGE_REVEAL = [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 255, 255]


def reveal_edges(subject, reveal_width):
    """Highlights the left, right, upper, and lower edges of an image
    with green, red, yellow, and cyan."""
    for side, color in zip(subject, _EDGE_REVEAL):
        view = side.view
        cols = side.edge.copy()
        rows = np.arange(view.shape[0])
        subtracts = [0] + ([1] * (reveal_width - 1))
        for n in subtracts:
            nz_cols = cols != 0
            cols[nz_cols] -= n
            view[rows, cols] = color
        bg = side.background
        bg = bg.reshape(bg.shape[0], bg.shape[2])
        view[::, 0] = bg


def combine_images(imgs, horizontal=True):
    axis = 1 if horizontal else 0
    combined = imgs[0]
    for img in imgs[1:]:
        combined = np.append(combined, img, axis=axis)
    return combined

