import json
import logging
from typing import List

import numpy as np

from recommendations.estimator import (
    DUMPS_FOLDER,
    Estimator,
)

log = logging.getLogger(__name__)


class OfflineJSONRecommender(Estimator):  # type: ignore
    """A class for an offline recommender that loads recommendations from a JSON file.

    This class extends the Estimator class, providing a fit method that does nothing
    and a recommend method that returns the recommendations loaded from the JSON file.
    """

    def __init__(self, dump: str, *args, **kwargs):
        """
        Args:
            dump: The name of the JSON file containing the recommendations to load.
        """
        super().__init__()
        with open(DUMPS_FOLDER.joinpath(dump), "r") as f:
            self.recommendations = json.load(f)

    def fit(self, df):
        """Overrides the fit method from the parent Estimator class.

        This method does nothing and simply logs a warning message.

        Args:
            df: The data to fit the recommender to. This argument is not used.

        Returns:
            The current instance of the class.
        """
        log.warning(f"Trying to fit {self.name} offline recommender")
        return self

    def recommend(self, users_ids: List[int], k: int = 10) -> np.ndarray:
        """Returns recommendations for the given users.

        Args:
            users_ids: A list of user ids to generate recommendations for.
            k: The number of recommendations to generate for each user.

        Returns:
            A numpy array containing the recommendations.

        Raises:
            AssertionError: If the length of users_ids is not 1.
        """
        assert len(users_ids) == 1, "Now only one user is supported"
        return np.array(self.recommendations[str(users_ids[0])][:k])
