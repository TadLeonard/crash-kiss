"""Miscellaneous tools for debugging edge finding, face smashing, etc"""

from six.moves import zip


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

