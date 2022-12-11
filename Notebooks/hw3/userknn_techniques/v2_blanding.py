from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender
from sklearn.preprocessing import MinMaxScaler


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


def bland_3(recs1, recs2, recs3, w1=0.7, w2=0.3, w3=0.1):
    # Create a scaler
    scaler = MinMaxScaler()

    # Scale the scores of the first two sets of recommendations
    for recs in [recs1, recs2]:
        recs["score_scaled"] = scaler.fit_transform(recs[["score"]]).reshape(
            -1, 1
        )

    # Merge the first two sets of recommendations
    merged = recs1.merge(
        recs2, on=["user_id", "item_id"], how="left", suffixes=["_1", "_2"]
    )

    # Fill in missing values with the corresponding values from the other set of recommendations
    merged["rank_2"].fillna(merged["rank_1"], inplace=True)
    merged["score_scaled_2"].fillna(merged["score_scaled_1"], inplace=True)

    # Compute the final score and ranking for the first two sets of recommendations
    merged["score"] = (
        merged["score_scaled_1"] * w1 + merged["score_scaled_2"] * w2
    )
    merged["rank"] = (
        merged.sort_values("score", ascending=False)
        .groupby("user_id")
        .cumcount()
        + 1
    )

    # Keep only the relevant columns
    merged = merged[["user_id", "item_id", "score", "rank"]]

    # Scale the scores of the third set of recommendations
    recs3["score_scaled"] = scaler.fit_transform(recs3[["score"]]).reshape(
        -1, 1
    )

    # Merge the third set of recommendations with the previous result
    merged = merged.merge(
        recs3, on=["user_id", "item_id"], how="left", suffixes=["_1", "_3"]
    )

    # Fill in missing values with the corresponding values from the previous result
    merged["rank_3"].fillna(merged["rank_1"], inplace=True)
    merged["score_scaled_3"].fillna(merged["score_scaled_1"], inplace=True)

    # Compute the final score and ranking for the merged recommendations
    merged["score"] = (
        merged["score_scaled_1"] * 0.9 + merged["score_scaled_3"] * w3
    )
    merged["rank"] = (
        merged.sort_values("score", ascending=False)
        .groupby("user_id")
        .cumcount()
        + 1
    )

    # Return the final result
    return merged[["user_id", "item_id", "rank"]]


POP_ITEMS_14D = [10440, 9728, 15297, 13865, 12192, 341, 4151, 3734, 12360, 7793]


class UserKNN_Blanding3:
    """Class for fit-perdict UserKNN model
    based on ItemKNN model from implicit.nearest_neighbours
    """

    def __init__(
        self,
        model1: ItemItemRecommender,
        model2: ItemItemRecommender,
        model3: ItemItemRecommender,
        pop_items: list,
        N_users: int = 50,
        w1: float = 0.7,
        w2: float = 0.3,
        w3: float = 0.1,
    ):
        self.N_users = N_users
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.is_fitted = False
        self.pop_items = pop_items

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def get_mappings(self, train):
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
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(
            train,
            users_mapping=self.users_mapping,
            items_mapping=self.items_mapping,
        )

        self.n = train.shape[0]
        self._count_item_idf(train)

        self.model1.fit(self.weights_matrix)
        self.model2.fit(self.weights_matrix)
        self.model3.fit(self.weights_matrix)

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

    def _get_recs(self, test, model, N):
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(
            model=model,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users,
        )

        recs = pd.DataFrame({"user_id": test["user_id"].unique()})
        recs["sim_user_id"], recs["sim"] = zip(*recs["user_id"].map(mapper))
        recs = (
            recs.set_index("user_id")
            .apply(pd.Series.explode)
            .reset_index()
            .drop_duplicates(["user_id", "sim_user_id"], keep=False)
            .merge(self.watched, on=["sim_user_id"], how="left")
            .explode("item_id")
            .sort_values(["user_id", "sim"], ascending=False)
            .drop_duplicates(["user_id", "item_id"], keep="first")
            .merge(
                self.item_idf, left_on="item_id", right_on="index", how="left"
            )
        )
        recs["score"] = recs["sim"] * recs["idf"]
        recs = recs.sort_values(["user_id", "score"], ascending=False)
        recs["rank"] = recs.groupby("user_id").cumcount() + 1
        recs = recs[recs["rank"] <= N]
        return recs

    def predict(self, test: pd.DataFrame, num_recs: int = 10):
        # get recommendations from each model
        recs1 = self._get_recs(test, self.model1, num_recs)
        recs2 = self._get_recs(test, self.model2, num_recs)
        recs3 = self._get_recs(test, self.model3, num_recs)

        # combine recommendations from all models
        recs = bland_3(recs1, recs2, recs3, self.w1, self.w2, self.w3)

        # keep only the top N recommendations for each user
        recs = recs[recs["rank"] <= num_recs][["user_id", "item_id", "rank"]]

        # group recommendations by user and convert to list of item IDs
        recommendations = recs.groupby("user_id").agg({"item_id": list})

        # combine recommendations with popular items
        recommendations["item_id"] = recommendations.loc[:, "item_id"].apply(
            lambda x: combine_recommendations_and_popular_items(
                np.array(x), np.array(self.pop_items), (10 - len(x))
            )
        )

        # explode list of items into separate rows
        prepared = recommendations.explode("item_id")
        # assign rank to each item for each user
        prepared["rank"] = prepared.groupby("user_id").cumcount() + 1
        # reset index
        prepared = prepared.reset_index()

        return prepared
