import os

import joblib
import numpy as np
import pandas as pd

from recommendations.estimator import (
    DUMPS_FOLDER,
    Estimator,
)

__all__ = ["LFM"]


class LFM(Estimator):  # type: ignore
    def __init__(self, name, *args, **kwargs):
        """
        Initialize the `LFM` class.

        Args:
            name (str): The name of the model.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        self.dataset = joblib.load(os.path.join(DUMPS_FOLDER, "dataset.pkl"))
        self.model = joblib.load(
            os.path.join(DUMPS_FOLDER, name.replace("zip", "pkl"))
        )

    def fit(self, df: pd.DataFrame) -> "LFM":
        """
        Fit the model to the given data.

        Args:
            df (pandas.DataFrame): The data to fit the model to.

        Returns:
            The fitted `LFM` instance.
        """
        return self

    def recommend(self, user_ids, n_recs=10):
        """
        Recommend items for the given user IDs.

        Args:
            user_ids (List[int]): The user IDs to recommend items for.
            n_recs (int, optional): The number of items to recommend. Defaults to 10.

        Returns:
            The recommended item IDs.
        """
        recs = self.model.recommend(
            np.array(user_ids), self.dataset, n_recs, filter_viewed=True
        )
        return recs


if __name__ == "__main__":
    lfm = LFM("LightFM.zip")
    print(lfm.recommend([1]))
