import os
import mahotas
import numpy as np
from crash_kiss import edge


img = mahotas.imread(os.path.join(os.path.dirname(__file__), "face.jpg"))
img = img[::, :-5:]  # make sure it's not a square to catch "wrong axis" bugs


def test_subject_default_sides():
    subj = edge.Subject(img=img)
    assert len(list(subj)) == 4

