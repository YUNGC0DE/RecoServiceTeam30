import numpy as np

from recommendations.estimator import Estimator


class ZeroModel(Estimator):  # type: ignore
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, df):
        return self

    def recommend(self, user_ids=None, k=10):
        return np.array(range(k))
