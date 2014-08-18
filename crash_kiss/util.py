"""Miscellaneous tools for debugging edge finding, face smashing, etc"""


def textify_edges(edges, char=u"@", scale=1.0):
    """Generates a text representation of an image subject's edges"""
    for edge_group in edges:
        prev_right = 0
        for left, right, _ in edge_group:
            if not right:
                continue
            yield u" " * int((left - prev_right) * scale)
            yield char * int((right - left) * scale)
            prev_right = right
        yield u"\n"


