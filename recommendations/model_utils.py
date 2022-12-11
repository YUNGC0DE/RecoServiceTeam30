import logging
import time
from functools import lru_cache
from typing import List

import numpy as np
import psutil

from recommendations.estimator import discover_models
from recommendations.resolvers import get_resolvers

log = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def load_model(model_name, *args, **kwargs):
    """Load model and initialize it. Cache it."""
    models = discover_models()
    if model_name not in models:
        raise ValueError(f"Unknown model `{model_name}`")
    start = time.time()
    rss_start = psutil.Process().memory_info().rss
    model = models[model_name](*args, **kwargs)
    log.info(
        f"Model `{model_name}` loaded in {time.time() - start:.2f} seconds,"
        f" RSS: {(psutil.Process().memory_info().rss - rss_start) / 1024 / 1024:.2f} Mb"
    )
    return model


def safe_recommend(
    model_name: str,
    user_ids: List[int],
    n_recs: int,
    resolution_strategy="pure_random",
):
    """
    Safe recommendation.

    Args:
        model_name: model name
        user_ids: user ids
        n_recs: number of recommendations
        resolution_strategy: resolution strategy if error occurs

    Returns:
        np.array: array of recommendations
    """

    resolvers = get_resolvers()

    if resolution_strategy not in resolvers:
        raise ValueError(f"Unknown resolution strategy `{resolution_strategy}`")

    try:
        model = load_model(model_name)
        recommendations = model.recommend(user_ids, n_recs)
        unique = []

        # Create a list of unique recommendations
        for rec in recommendations:
            if rec not in unique:
                unique.append(rec)

        # If there are not enough unique recommendations, use the resolution strategy
        # to get additional recommendations
        if len(unique) < n_recs:
            resolved = resolvers[resolution_strategy](user_ids, n_recs)
            for x in resolved:
                if x not in unique:
                    unique.append(x)
                    # Stop when we have enough unique recommendations
                    if len(unique) == n_recs:
                        break

        # Return the first n_recs unique recommendations
        return np.array(unique[:n_recs])
    except Exception as e:
        log.error(e)
        return resolvers[resolution_strategy](user_ids, n_recs)
