import re
import numpy as np
import pandas as pd

def apply_regex_series(series: pd.Series, regex: re.Pattern, handle_nan=True):
    """
    Applies compiled regex to pandas Series.

    Args:
        series (pd.Series): Series to apply regex
        regex (re.Pattern): Regex pattern compiled.
        handle_nan (bool): True if you want to return 0 when
        value is nan, otherwise nan is passed to regex.
    Returns:
        np.array: Numpy array containing 1 for every match
        and 0 for no match.
    """

    if handle_nan:
        return (np.where(
            series.str.contains(regex) & series.notna(), 1, 0))
    else:
        return (np.where(
            series.str.contains(regex), 1, 0))