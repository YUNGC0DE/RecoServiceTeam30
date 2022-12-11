import time
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
)

import joblib
import numpy
import numpy as np
import pandas as pd
from implicit.nearest_neighbours import ItemItemRecommender

from recommendations.estimator import (
    DUMPS_FOLDER,
    Estimator,
)
from recommendations.model_utils import load_model

__all__ = ["UserKNN"]


from typing import (
    List,
    Union,
)

from numpy import (
    array,
    ndarray,
)


def merge_sorted_unique(items: Union[ndarray, List[List]], n: int) -> ndarray:
    """
    Merge the elements from a list of sorted lists into a single list with unique elements.

    Args:
        items: A 2D numpy array of sorted integers.
        n: The maximum number of elements to include in the output array.

    Returns:
        A 1D numpy array containing the first `n` unique elements from the input array, in the order in which they were
        encountered.
    """
    # Initialize the output array
    result = []

    # Iterate over the input lists
    for item in items:
        # Iterate over the elements in the current list
        for j in item:
            # If the current element is not in the output array, add it
            if j not in result:
                result.append(j)
                # If the output array has reached the maximum length, return it
                if len(result) == n:
                    return array(result)

    # Return the output array
    return array(result)


class UserKNN(Estimator):  # type: ignore
    """A user-based KNN model for recommendation systems.

    This class implements a user-based KNN model for recommending items to users.
    It uses the `implicit` library to compute the similarity between users, and
    provides a method for generating recommendations for a given user.

    Attributes:
        model: The `ItemItemRecommender` instance used to compute the similarity
            between users.
        N_users: The maximum number of similar users to consider when generating
            recommendations.
        use_idf_weights: A flag indicating whether to use IDF weights when
            sorting items for each user.
        mapper: A callable that generates recommendations for a given user.
        users_mapping: A mapping from external user IDs to internal IDs used by
            the `ItemItemRecommender` instance.
        items_mapping: A mapping from external item IDs to internal IDs used by
            the `ItemItemRecommender` instance.
        users_inv_mapping: The inverse mapping of `users_mapping`, used to
            convert internal user IDs to external IDs.
        items_inv_mapping: The inverse mapping of `items_mapping`, used to
            convert internal item IDs to external IDs.
        weights: A dictionary mapping item IDs to their IDF weights, if
            `use_idf_weights` is `True`.
        train: A dictionary mapping user IDs to a list of items, sorted by their
            IDF weights if `use_idf_weights` is `True`, otherwise the items are
            not sorted.

    """

    def __init__(
        self,
        model: Union[ItemItemRecommender, Path],
        users_mapping_dump: str,
        items_mapping_dump: str,
        train: str,
        N_users: int = 50,
        use_idf_weights: bool = True,
        *args,
        **kwargs,
    ):

        """Initializes the `UserKNN` instance.

        Args:
            model: The `ItemItemRecommender` instance to use for computing the
                similarity between users, or a `Path` object pointing to a
                `joblib` dump of the `ItemItemRecommender` instance.
            users_mapping_dump: The path to a `joblib` dump of a dictionary
                mapping external user IDs to internal IDs.
            items_mapping_dump: The path to a `joblib` dump of a dictionary
                mapping external item IDs to internal IDs.
            train: The path to the CSV file containing the training data.
            N_users: The maximum number of similar users to consider when
                generating recommendations (default: 50).
            use_idf_weights: A flag indicating whether to use IDF weights when
                sorting items for each user (default: `True`).
            *args: Additional arguments to pass to the parent class constructor.
            **kwargs: Additional keyword arguments to pass to the parent class
        """

        super().__init__()

        if isinstance(model, ItemItemRecommender):
            self.model = model
        else:
            self.model = joblib.load(DUMPS_FOLDER.joinpath(model))

        self.N_users = N_users
        self.use_idf_weights = use_idf_weights
        self.mapper = self._generate_recommendations_mapper(N_users)

        self.users_mapping: Dict[int, int] = joblib.load(
            DUMPS_FOLDER.joinpath(users_mapping_dump)
        )
        self.items_mapping: Dict[int, int] = joblib.load(
            DUMPS_FOLDER.joinpath(items_mapping_dump)
        )
        self.users_inv_mapping: Dict[int, int] = {
            v: k for k, v in self.users_mapping.items()
        }
        self.items_inv_mapping: Dict[int, int] = {
            v: k for k, v in self.items_mapping.items()
        }

        interactions = pd.read_csv(
            DUMPS_FOLDER.joinpath(train), usecols=["user_id", "item_id"]
        )

        self.weights: Optional[Dict[int, float]] = (
            self._get_idf_weights(interactions)
            if self.use_idf_weights
            else None
        )
        self.train = self._load_train(interactions)

    def _load_train(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        """Load the training data from the input dataframe.

        Args:
            df: The input dataframe with the training data.
                This dataframe should have columns 'user_id' and 'item_id'.

        Returns:
            A dictionary mapping user_ids to lists of item_ids, sorted by weight in descending order.
        """
        df["rank"] = df.groupby("user_id").cumcount() + 1

        # Subset the dataframe to only include the first N_users items for each user
        rows = df[df["rank"] <= self.N_users]

        # Convert the dataframe to numpy arrays
        user_ids = rows["user_id"].to_numpy()
        item_ids = rows["item_id"].to_numpy()

        recommendations: Dict[int, List[int]] = {}

        # Loop through the user_ids and item_ids and add the
        # items to the recommendations dictionary
        for user_id, item_id in zip(user_ids, item_ids):
            recommendations.setdefault(user_id, []).append(item_id)

        # If weights (IDF) are being used,
        # sort the items in each user's list of recommendations by weight in descending order
        if self.weights:
            for user_id, items in recommendations.items():
                recommendations[user_id] = sorted(
                    items, key=lambda x: self.weights[x], reverse=True  # type: ignore
                )

        return recommendations

    def _get_idf_weights(self, df: pd.DataFrame) -> Dict[int, float]:
        # Use numpy arrays to store the data
        item_ids = df["item_id"].to_numpy()

        # Calculate the number of times each item appears in the data
        unique_item_ids, counts = np.unique(item_ids, return_counts=True)

        # Calculate the IDF weights for each item
        n = df.shape[0]
        weights = np.log((1 + n) / (1 + counts) + 1)

        # Create a dictionary mapping item_ids to weights
        weights_dict = {}
        for item_id, weight in zip(unique_item_ids, weights):
            weights_dict[item_id] = weight

        return weights_dict

    def _generate_recommendations_mapper(self, N: int) -> Callable:
        def _mapper(user: int) -> Tuple[List[int], List[int]]:
            user_id = self.users_mapping.get(user, None)

            if user_id is None:
                return [], []

            # Get the top N similar users
            recommendations = self.model.similar_items(user_id, N=N)

            return (
                [
                    self.users_inv_mapping[user_]
                    for user_, _ in recommendations
                    if user_ != user_id
                ],
                [
                    similar
                    for user_, similar in recommendations
                    if user_ != user_id
                ],
            )

        return _mapper

    def recommend(self, user_ids: List[int], N: int) -> np.ndarray:
        # Initialize an empty list to store the recommendations
        recommendations = []

        # Loop through the user ids
        for user_id in user_ids:
            # Get the similar user ids and scores for the user
            sim_user_ids, scores = self.mapper(user_id)

            # Get the top N similar users
            sim_user_ids = sim_user_ids[: self.N_users]

            # Get the items from the training data for the top N similar users
            sim_user_items = np.array(
                [
                    np.array(self.train[sim_user_id])
                    for sim_user_id in sim_user_ids
                ],
                dtype=object,
            )

            # Merge the items from the similar users and sort them in descending order by score
            recs = merge_sorted_unique(sim_user_items, N)

            # Add the recommendations to the list
            recommendations.append(recs)

        # TODO (qnbhd): elliminate for more common case
        if len(recommendations) == 1:
            recommendations = recommendations[0]  # type: ignore

        # Return the recommendations as a numpy array
        return numpy.array(recommendations)


if __name__ == "__main__":
    start = time.time()
    model = load_model("userknn-bm25-online")
    recos = model.recommend([999], 10)
    print(recos)
    print("-- %s seconds --" % (time.time() - start))
