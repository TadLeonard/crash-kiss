from six.moves import zip
import numpy as np


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


