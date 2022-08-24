import bisect

def find_le(ls: list, x: int):
    """
    Receives ordered list a and element x and rightmost
    value less than or equal to x.
    Args:
        ls (list): ordered list of int
        x (int): element to find
    """

    i = bisect.bisect_right(ls, x)
    if i:
        return ls[i-1]
    raise ValueError