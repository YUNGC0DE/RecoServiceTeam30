from pathlib import Path

import joblib
import pandas as pd
from rectools.dataset import Dataset

from recommendations.estimator import (
    DUMPS_FOLDER,
    Estimator,
)

__all__ = ["Popular"]


class Popular(Estimator):  # type: ignore

    def __init__(self, model_dump, dataset_dump, *args, **kwargs):
        """
        Initialize the `Popular` class.

        Args:
            model_dump (str): The file path of the dump for the model.
            dataset_dump (str): The file path of the dump for the dataset.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        dump_model_path = Path(model_dump)
        dataset_path = Path(dataset_dump)

        # Load the dataset and model from their respective dump files
        self.rectools_dataset: Dataset = joblib.load(
            DUMPS_FOLDER.joinpath(dataset_path)
        )
        self.model = joblib.load(DUMPS_FOLDER.joinpath(dump_model_path))

    def fit(self, df: pd.DataFrame) -> "Popular":
        """
        Fit the model to the given data.

        Args:
            df (pandas.DataFrame): The data to fit the model to.

        Returns:
            The fitted `Popular` instance.
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
        predictions: pd.Series = self.model.recommend(
            user_ids,
            dataset=self.rectools_dataset,
            k=n_recs,
            filter_viewed=False,
        )
        return predictions["item_id"].values
