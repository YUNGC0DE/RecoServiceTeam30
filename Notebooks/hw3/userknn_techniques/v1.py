from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender


def get_top_k_items_for_last_n_days(
    df: pd.DataFrame, k: int = 10, days: int = 14, all_time: bool = False
) -> np.array:
    """
    Return a np.array of top@k most popular items for last N days
    """
    if all_time is True:
        recommendations = (
            df.loc[:, "item_id"].value_counts().head(k).index.values
        )
    else:
        min_date = df["datetime"].max().normalize() - pd.DateOffset(days)
        recommendations = (
            df.loc[df["datetime"] > min_date, "item_id"]
            .value_counts()
            .head(k)
            .index.values
        )
    return list(recommendations)


def combine_recommendations_and_popular_items(
    recommendations: np.array, popular_items: np.array, number: int
) -> list:
    """
    Add number of items from popular_items to recommendations.
    Return unique array of 10 items
    """
    K = 10

    size_of_array = np.unique(recommendations).size
    if size_of_array == K:
        return recommendations

    def remove_duplicate_items(mix_pop: np.array, k: int = K) -> np.array:
        """
        Delete all duplicates items in array
        """

        size_of_array = np.unique(mix_pop).size
        if size_of_array == K:
            return mix_pop
        else:
            # delete duplicates and save the order of items
            mix_pop = mix_pop[np.sort(np.unique(mix_pop, return_index=True)[1])]
            # add new items from popular_items
            i = k - size_of_array
            mix_pop = np.concatenate(
                (mix_pop, np.random.choice(popular_items, i, replace=False))
            )
            return remove_duplicate_items(mix_pop)

    full_arr_mix_pop = np.concatenate((recommendations, popular_items[:number]))
    full_arr_mix_pop = remove_duplicate_items(full_arr_mix_pop)

    return list(full_arr_mix_pop)


POP_ITEMS_14D = [10440, 9728, 15297, 13865, 12192, 341, 4151, 3734, 12360, 7793]


class my_UserKnn:
    def __init__(
        self, model: ItemItemRecommender, pop_items: list, N_users: int = 50
    ):
        self.N_users = N_users
        self.model = model
        self.is_fitted = False
        self.pop_items = pop_items

    def set_mappings(self, train):
        self.users_inv_mapping = dict(enumerate(train["user_id"].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train["item_id"].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def get_matrix(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        weight_col: str = None,
        users_mapping: Dict[int, int] = None,
        items_mapping: Dict[int, int] = None,
    ):

        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        interaction_matrix = sp.sparse.coo_matrix(
            (
                weights,
                (
                    df[user_col].map(self.users_mapping.get),
                    df[item_col].map(self.items_mapping.get),
                ),
            )
        )

        self.watched = df.groupby(user_col).agg({item_col: list})
        return interaction_matrix

    def idf(self, n: int, x: float):
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: pd.DataFrame):
        item_cnt = Counter(df["item_id"].values)
        item_idf = pd.DataFrame.from_dict(
            item_cnt, orient="index", columns=["doc_freq"]
        ).reset_index()
        item_idf["idf"] = item_idf["doc_freq"].apply(
            lambda x: self.idf(self.n, x)
        )
        self.item_idf = item_idf

    def fit(self, train: pd.DataFrame):
        self.user_knn = self.model
        self.set_mappings(train)
        self.weights_matrix = self.get_matrix(
            train,
            users_mapping=self.users_mapping,
            items_mapping=self.items_mapping,
        )

        self.n = train.shape[0]
        self._count_item_idf(train)

        self.user_knn.fit(self.weights_matrix)
        self.is_fitted = True

    def _generate_recs_mapper(
        self,
        model: ItemItemRecommender,
        user_mapping: Dict[int, int],
        user_inv_mapping: Dict[int, int],
        N: int,
    ):
        def _recs_mapper(user):
            user_id = user_mapping[user]
            recs = model.similar_items(user_id, N=N)
            return [user_inv_mapping[user] for user, _ in recs], [
                sim for _, sim in recs
            ]

        return _recs_mapper

    def predict(self, test: pd.DataFrame, N_recs: int = 10):

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        # Generate the recommendations mapper
        mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users,
        )

        # Create a DataFrame of the unique user IDs from the test set
        recs = pd.DataFrame({"user_id": test["user_id"].unique()})

        # Add the recommended user IDs and similarities to the DataFrame
        recs["sim_user_id"], recs["sim"] = zip(*recs["user_id"].map(mapper))

        # Explode the DataFrame so that each row corresponds to a single recommended user
        recs = recs.set_index("user_id").apply(pd.Series.explode).reset_index()

        # Filter out recommendations for the user's own ratings
        recs = recs[~(recs["user_id"] == recs["sim_user_id"])]

        # Merge the recommended user IDs with the watched items DataFrame
        recs = recs.merge(
            self.watched,
            left_on=["sim_user_id"],
            right_on=["user_id"],
            how="left",
        )

        # Explode the list of items watched by each recommended user
        recs = recs.explode("item_id")

        # Sort the DataFrame by user and similarity score
        recs = recs.sort_values(["user_id", "sim"], ascending=False)

        # Keep only the first recommendation for each user-item pair
        recs = recs.drop_duplicates(["user_id", "item_id"], keep="first")

        # Merge the DataFrame with the IDF values for each item
        recs = recs.merge(
            self.item_idf, left_on="item_id", right_on="index", how="left"
        )

        # Calculate the score for each recommendation
        recs["score"] = recs["sim"] * recs["idf"]

        # Sort the DataFrame by user and score
        recs = recs.sort_values(["user_id", "score"], ascending=False)

        # Add a rank column to the DataFrame
        recs["rank"] = recs.groupby("user_id").cumcount() + 1

        # Filter out recommendations with a rank higher than N_recs
        recs = recs[recs["rank"] <= N_recs]

        # Group the DataFrame by user and aggregate the item IDs into a list
        recommendations = recs.groupby("user_id").agg({"item_id": list})

        # Use the combine_recommendations_and_popular_items function to combine the recommendations
        # with the most popular items

        recommendations["item_id"] = recommendations.loc[:, "item_id"].apply(
            lambda x: combine_recommendations_and_popular_items(
                np.array(x), np.array(self.pop_items), (N_recs - len(x))
            )
        )

        prepared = recommendations.explode("item_id")
        prepared["rank"] = prepared.groupby("user_id").cumcount() + 1
        prepared = prepared.reset_index()

        return prepared
