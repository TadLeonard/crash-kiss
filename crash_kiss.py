from __future__ import print_function
from collections import namedtuple
import argparse
import mahotas
import numpy as np


DEFAULT_THRESH = 15
DEFAULT_NEG_SAMPLE = 15


parser = argparse.ArgumentParser(description="Crash images into things")
parser.add_argument("target")
parser.add_argument("--output", default=None)
parser.add_argument("--threshold", default=DEFAULT_THRESH, type=int)
parser.add_argument("--smash", type=str,
                    choices=("left", "right", "center"), default=None)
parser.add_argument("--output-type", type=str,
                    choices=("ascii", "gif", "img"), default="ascii")


def wall_smash_image(img_edges, shift_right=True):
    """Mutates a numpy array of an image so that the subject
    is smashed to the right or left edge."""
    for left, right, negspace in img_edges:
        rowlen = right - left
        if not rowlen:
            continue
        row[-rowlen:] = row[left: right]
        row[:-rowlen] = negspace


def get_ascii_edges(img_edges, char=u"@"):
    return "\n".join((u" " * left) + (char * (right - left))
                     for left, right, _ in img_edges)


row_data = namedtuple("row_data", "left_idx right_idx neg_space")


def iter_edge_indices(
        img, neg_sample=DEFAULT_NEG_SAMPLE, threshold=DEFAULT_THRESH):
    """Finds the edges of the subject for each row of pixels. This assumes that
    the subject is on a background with significantly different RGB values.
    Yields `row_data` instances."""
    for row in img:
        neg_space = np.mean(pixel_row[:neg_sample], axis=0)
        pos_space = np.all(np.abs(pixel_row - neg_space) > threshold, axis=1)
        left_edge = np.argmax(pos_space)
        right_edge = np.argmax(pos_space[::-1])
        if right_edge:
            width = pixel_row.shape[0]
            right_edge = width - right_edge
        yield row_data(left_edge, right_edge, neg_space)


def run():
    args = parser.parse_args()
    img = mahotas.imread(args.target)
    im2 = img[::, ::-1]
    mahotas.imsave("floop", im2)
    if args.smash:
        if args.smash == "left":
            raise NotImplementedError("Can't smash left yet")
        elif args.smash == "right":
            wall_smash_image(img)
        elif args.smash == "center":
            raise NotImplementedError("Can't smash center yet")
    mahotas.imsave("bleeeeup", img)
    print(get_ascii_edges(img, char=u"@"))


if __name__ == "__main__":
    run()
