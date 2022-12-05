import inspect
import sys

import numpy as np


def topfixed(user_id, n_recs: int, max_index: int = 1_000_000) -> np.ndarray:
    return np.array(
        [9728, 15297, 10440, 14488, 13865, 12192, 341, 4151, 3734, 512][:n_recs]
    )


def get_resolvers():
    """
    Get all resolvers.

    Returns:
        dict: dict of resolvers
    """
    return {
        name: obj
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isfunction(obj) and name != "get_resolvers"
    }
