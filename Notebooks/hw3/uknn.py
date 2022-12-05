from collections import Counter
from pathlib import Path
from typing import (
    Dict,
    List,
    Union,
)

import joblib
import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender


def popoular_number_of_items_days(
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


def full_reco_items_list(
    arr_reco_after_model: np.array, pop_array: np.array, number: int
) -> list:
    """
    Add number of items from pop_array to arr_reco_after_model.
    Return unique array of 10 items
    """

    CONST_K = 10

    size_of_array = np.unique(arr_reco_after_model).size
    if size_of_array == CONST_K:
        return arr_reco_after_model

    # all duplicates will be deleting and adding some items from pop_array
    def del_repeat_items(
        full_arr_mix_pop: np.array, k: int = CONST_K
    ) -> np.array:
        """
        Delete all duplicates items in array
        """

        size_of_array = np.unique(full_arr_mix_pop).size
        if size_of_array == CONST_K:
            return full_arr_mix_pop
        else:
            # delete duplicates and save the order of items
            full_arr_mix_pop = full_arr_mix_pop[
                np.sort(np.unique(full_arr_mix_pop, return_index=True)[1])
            ]
            # add new items from pop_array
            i = k - size_of_array
            full_arr_mix_pop = np.concatenate(
                (
                    full_arr_mix_pop,
                    np.random.choice(pop_array, i, replace=False),
                )
            )
            return del_repeat_items(full_arr_mix_pop)

    full_arr_mix_pop = np.concatenate(
        (arr_reco_after_model, pop_array[:number])
    )
    full_arr_mix_pop = del_repeat_items(full_arr_mix_pop)

    return list(full_arr_mix_pop)


POP_ITEMS_14D = [10440, 9728, 15297, 13865, 12192, 341, 4151, 3734, 12360, 7793]


class UserKnnv2:
    """Class for fit-perdict UserKNN model
    based on ItemKNN model from implicit.nearest_neighbours
    """

    name = "userknn"

    def __init__(
        self,
        model_path=None,
        train_path=None,
        n=50,
    ):

        model_path = model_path or Path(__file__).parent.joinpath(
            "bm25-userknn-v2.joblib"
        )
        train_path = train_path or Path(__file__).parent.joinpath(
            "interactionsv2.csv"
        )
        if not model_path.exists():
            raise ValueError(f"Model dump not found at {model_path}")
        if not train_path.exists():
            raise ValueError(f"Train dataset dump not found at {train_path}")

        self.N_users = n
        self.model = joblib.load(model_path)
        self.list_pop_items_14d = POP_ITEMS_14D

        self.fit(pd.read_csv(train_path))

        self.mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users,
        )

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
        self.user_knn = self.model
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(
            train,
            users_mapping=self.users_mapping,
            items_mapping=self.items_mapping,
        )
        self.n = train.shape[0]
        self._count_item_idf(train)

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

    def recommend(
        self, test: Union[List[int], np.ndarray, pd.DataFrame], N_recs: int = 10
    ):

        if isinstance(test, pd.DataFrame):
            iterable = test["user_id"].unique()
        elif isinstance(test, np.ndarray):
            iterable = np.array(set(test))
        elif isinstance(test, list):
            # only for test
            iterable = test
        else:
            raise ValueError("test must be pd.DataFrame or np.ndarray or list")

        recs = pd.DataFrame({"user_id": iterable})
        recs["sim_user_id"], recs["sim"] = zip(
            *recs["user_id"].map(self.mapper)
        )

        recs = recs.set_index("user_id").apply(pd.Series.explode).reset_index()

        recs = (
            recs[~(recs["user_id"] == recs["sim_user_id"])]
            .merge(
                self.watched,
                left_on=["sim_user_id"],
                right_on=["user_id"],
                how="left",
            )
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
        recs = recs[recs["rank"] <= N_recs]

        final_reco = recs.groupby("user_id").agg({"item_id": list})

        final_reco["item_id"] = final_reco.loc[:, "item_id"].apply(
            lambda x: full_reco_items_list(
                np.array(x), np.array(self.list_pop_items_14d), (10 - len(x))
            )
        )
        # используется только в сервисе или для проверки
        reco_for_user = final_reco.loc[iterable, "item_id"]

        # # преобразование таблицы для метрик, только для колаба
        # my_reco = final_reco.explode('item_id')
        # my_reco['rank'] = my_reco.groupby('user_id').cumcount() + 1
        # my_reco = my_reco.reset_index()

        # return list(reco_for_user.iloc[0]) # для сервиса
        return reco_for_user
