import time
from pathlib import Path
from typing import (
    Dict,
    List,
    Union,
)

import joblib
import numpy as np
from lightfm import LightFM

from recommendations.estimator import (
    DUMPS_FOLDER,
    Estimator,
)
from recommendations.model_utils import load_model


class LFMOnline(Estimator):  # type: ignore
    """
    A class for making online recommendations using a trained LightFM model.

    Attributes:
        model: The `LightFM` instance used for making recommendations.
        users_mapping: A mapping from external user IDs to internal IDs used by
            the `ItemItemRecommender` instance.
        items_mapping: A mapping from external item IDs to internal IDs used by
            the `ItemItemRecommender` instance.
        users_inv_mapping: The inverse mapping of `users_mapping`, used to
            convert internal user IDs to external IDs.
        items_inv_mapping: The inverse mapping of `items_mapping`, used to
            convert internal item IDs to external IDs.
        users_biases: The biases of the users in the model.
        users_embeddings: The embeddings of the users in the model.
        items_embeddings: The embeddings of the items in the model.
    """

    def __init__(
        self,
        model: Union[LightFM, Path],
        users_mapping_dump: str,
        items_mapping_dump: str,
        *args,
        **kwargs,
    ):
        """
        Initialize the `LFMOnline` class.

        Args:
            model (Union[LightFM, Path]): The trained LightFM model to use for recommendations,
                provided either as a `LightFM` object or a path to a `.joblib` file containing
                the model.
            users_mapping_dump (str): The path to the `.joblib` file containing the user mapping
                dictionary. This dictionary maps original user IDs to the internal IDs used by the
                model.
            items_mapping_dump (str): The path to the `.joblib` file containing the item mapping
                dictionary. This dictionary maps original item IDs to the internal IDs used by the
                model.
        """
        # Initialize parent class
        super().__init__()

        # Load LightFM model from path or use provided model
        if isinstance(model, LightFM):
            self.model = model
        else:
            self.model = joblib.load(DUMPS_FOLDER.joinpath(model))

        # Load mappings
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

        # Get user and item embeddings
        user_representations = self.model.get_user_representations()
        self.users_biases = user_representations[0]
        self.users_embeddings = user_representations[1]

        # Get item representations
        item_representations = self.model.get_item_representations()
        items_biases = item_representations[0]
        items_embeddings = item_representations[1]

        # Only use a subset of the items' biases and embeddings
        items_embedding = items_embeddings[
            : len(self.items_inv_mapping.keys()), :
        ]
        items_biases = items_biases[: len(self.items_inv_mapping.keys())]

        # Concatenate biases, a column of ones, and the item embeddings
        self.items_embeddings = np.hstack(
            (
                np.ones((items_biases.size, 1)),
                items_biases[:, np.newaxis],
                items_embedding,
            )
        )

    def recommend(self, user_ids: List[int], n: int) -> np.ndarray:
        """
        Recommend items for the given user.

        Args:
            user_ids (list): A list of user IDs for which to recommend items.
            n_recs (int): The number of items to recommend for each user.

        Returns:
            items_to_recommend (list): A list of item IDs that are recommended for each user.
        """

        inner_id = self.users_mapping[user_ids[0]]

        # Get the user's embedding
        user_embedding = np.hstack(
            (
                self.users_biases[inner_id],
                np.ones(self.users_biases[inner_id].size),
                self.users_embeddings[inner_id],
            ),
        )

        # Compute scores by taking the dot product of the user's embedding with each item's embedding
        scores = np.dot(self.items_embeddings, user_embedding)

        # Get the indices of the top `n_recs` scores
        # fast top-k via argpartition
        top_score_ids = np.argpartition(scores, -n)[-n:][::-1]

        # Get the items corresponding to the top score indices
        items_to_recommend = np.array(
            [
                self.items_inv_mapping[item]
                for item in top_score_ids
                if item in self.items_inv_mapping
            ]
        )

        return np.array(items_to_recommend)


if __name__ == "__main__":
    model = load_model("lightfm-online")
    start = time.time()
    recos = model.recommend([1], 10)
    print(recos)
    print("-- %s seconds --" % (time.time() - start))
