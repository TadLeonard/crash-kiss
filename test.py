import os
import mahotas
import numpy as np
from crash_kiss import edge


def _get_test_img():
    img = mahotas.imread(os.path.join(os.path.dirname(__file__), "face.jpg"))
    # cut off bottom non-white edge
    # make sure it's not square to catch "wrong axis" bugs
    img = img[:-10:, :-7:]
    # make the top rows of the image pure white
    img[:5] = [255, 255, 255]
    return img


def test_subject_default_sides():
    img = _get_test_img()
    subj = edge.Subject(img=img)
    assert len(list(subj)) == 4


def test_test_image():
    """Make sure our test image, face.jpg, is white on the edges"""
    img = _get_test_img()
    all = np.all(img[::, :10:] > 240, axis=2)
    assert np.all(all)


def test_subject_config():
    s = edge.Subject(config=edge.config(threshold=70, bg_change_tolerance=55))
    assert s._config["threshold"] == 70
    assert s._config["bg_change_tolerance"] == 55


def test_side_config():
    s = edge.Subject(config=edge.config(threshold=70, bg_change_tolerance=55))
    assert s.left._config["threshold"] == 70
    assert s.left._config["bg_change_tolerance"] == 55
    c = s._config
    assert all(side._config == c for side in s)
    

def test_edge_cleaning():
    """Make sure that rows that are all background (i.e. all white) do not
    get an edge index of zero but are instead masked."""
    img = _get_test_img()
    assert np.all(img[:5] == [255, 255, 255])  # top rows aughtta be white
    sub = edge.Subject(img=img)
    # Edges are masked arrays. A True value in a masked array indicates that
    # the value is masked, so we want white rows's edges to be masked
    # or == True
    assert np.all(sub.left.edge[:5].mask == True)
    

def test_edge_below_threshold():
    img = _get_test_img()
    img[::, 4:6] = [230, 230, 230]
    config = edge.config(threshold=60)
    sub = edge.Subject(img=img, config=config)
    print sub.left.edge[:-10]
    assert np.all(sub.left.edge >= 5)
    

def test_edge_above_threshold():
    img = _get_test_img()
    img[:8:, 4:6:] = [10, 10, 10]
    config = edge.config(threshold=60)
    subj = edge.Subject(img=img, config=config)
    l_edge = subj.left.edge
    assert np.all(l_edge >= 4)

