import bisect

def find_le(ls, x):
    '''Receives ordered list a and element x.
    Returns rightmost value less than or equal to x.'''

    i = bisect.bisect_right(ls, x)
    if i:
        return ls[i-1]
    raise ValueError