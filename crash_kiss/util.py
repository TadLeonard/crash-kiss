"""Miscellaneous tools for debugging edge finding, face smashing, etc"""

from six.moves import zip


def textify_edges(edges, char=u"@", scale=1.0):
    """Generates a text representation of an image subject's edges"""
    prev_right = 0
    for left, right in zip(*edges):
        if not right:
            continue
        yield u" " * int((left - prev_right) * scale)
        yield char * int((right - left) * scale)
        prev_right = right
    yield u"\n"


