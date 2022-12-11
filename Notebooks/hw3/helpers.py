from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)

import dask.dataframe as dd
import numpy as np
import pandas as pd
from implicit.nearest_neighbours import ItemItemRecommender
from rectools.metrics import calc_metrics
from rectools.model_selection import TimeRangeSplitter
from scipy.sparse import coo_matrix

PathLike = Union[str, Path]


class RecommendationsDatasets(NamedTuple):
    """Named tuple to hold the datasets used for recommendations"""

    interactions: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame

    @classmethod
    def from_raw_data(
        cls,
        interactions: pd.DataFrame,
        users: pd.DataFrame,
        items: pd.DataFrame,
    ) -> "RecommendationsDatasets":
        """Create a RecommendationsDatasets object from raw data"""
        interactions.rename(
            columns={"last_watch_dt": "datetime", "total_dur": "weight"},
            inplace=True,
        )
        interactions["datetime"] = pd.to_datetime(interactions["datetime"])
        return RecommendationsDatasets(
            interactions=interactions, users=users, items=items
        )


class Mappings(NamedTuple):
    """Named tuple to hold the mappings"""

    direct: dict
    reverse: dict


def load_datasets(path: PathLike) -> RecommendationsDatasets:
    """Load the datasets used for recommendations"""
    path = Path(path).resolve()
    interactions = pd.read_csv(path / "interactions.csv")
    users = pd.read_csv(path / "users.csv")
    items = pd.read_csv(path / "items.csv")
    return RecommendationsDatasets.from_raw_data(interactions, users, items)


def get_crossval_splitter(
    last_date: pd.Timestamp, n_folds: int, n_units: int, unit: str
) -> TimeRangeSplitter:
    """Get the cross-validation splits"""
    start_date = last_date - pd.Timedelta(n_folds * n_units + 1, unit=unit)

    date_range = pd.date_range(
        start=start_date,
        periods=n_folds + 1,
        freq=f"{n_units}{unit}",
        tz=last_date.tz,
    )

    return TimeRangeSplitter(
        date_range=date_range,
        filter_already_seen=True,
        filter_cold_items=True,
        filter_cold_users=True,
    )


def get_mappings(train_dataset: pd.DataFrame, column: str) -> Mappings:
    """Get the mappings for a given column"""
    reversed = dict(enumerate(train_dataset[column].unique()))
    mapping = {v: k for k, v in reversed.items()}
    return Mappings(direct=mapping, reverse=reversed)


def get_implicit_recommendations_mapper(
    model: ItemItemRecommender,
    N: int,
    user_mapping: Mappings,
) -> Callable[[int], Tuple[List, List]]:
    """Get the mapper for implicit recommendations"""

    def _mapper(user) -> Tuple[List, List]:
        user_id = user_mapping.direct[user]
        recommendations = model.similar_items(user_id, N=N)
        return (
            [user_mapping.reverse[user] for user, _ in recommendations],
            [sim for _, sim in recommendations],
        )

    return _mapper


def join_watched(
    recommendations_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    use_dask: bool = False,
) -> pd.DataFrame:
    """Join the watched items to the recommendations"""
    recs_df = recommendations_dataframe
    train_df = train_dataframe

    if use_dask:
        recs_df = dd.from_pandas(recommendations_dataframe, npartitions=10)
        train_df = dd.from_pandas(train_dataframe, npartitions=10)

    watched = train_df.groupby("user_id").agg({"item_id": list})
    recs_df["similar_user_id"] = recs_df["similar_user_id"].astype("int64")
    recs_df = recs_df.merge(
        watched, left_on=["similar_user_id"], right_on=["user_id"], how="left"
    )
    recs_df = recs_df.explode("item_id")
    recs_df = recs_df.drop_duplicates(["user_id", "item_id"], keep="first")
    recs_df = recs_df.sort_values(["user_id", "similarity"], ascending=False)
    return recs_df.compute() if use_dask else recs_df


def rank_from_sim(
    recommendations_dataframe: pd.DataFrame, train_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """Rank the recommendations from the similarity"""
    n = train_dataframe.shape[0]
    idf = pd.DataFrame(
        train_dataframe["item_id"].value_counts(), columns=["doc_freq"]
    ).reset_index()
    idf["idf"] = idf["doc_freq"].apply(lambda x: np.log((1 + n) / (1 + x) + 1))
    recommendations_dataframe = recommendations_dataframe.merge(
        idf[["index", "idf"]],
        left_on=["item_id"],
        right_on=["index"],
        how="left",
    ).drop(columns=["index"])
    recommendations_dataframe["rank_idf"] = (
        recommendations_dataframe["similarity"]
        * recommendations_dataframe["idf"]
    )
    recommendations_dataframe.sort_values(
        ["user_id", "rank_idf"], ascending=False, inplace=True
    )
    recommendations_dataframe["rank"] = (
        recommendations_dataframe.groupby("user_id").cumcount() + 1
    )
    return recommendations_dataframe


def get_coo_mat(
    dataframe: pd.DataFrame,
    users_mapping: Mappings,
    items_mapping: Mappings,
    user_column: str = "user_id",
    item_column: str = "item_id",
    weights_column: Optional[str] = None,
) -> coo_matrix:
    """Create a coo_matrix from a dataframe"""
    if weights_column is None:
        weights = np.ones(dataframe.shape[0])
    else:
        weights = dataframe[weights_column].values
    return coo_matrix(
        (
            weights,
            (
                dataframe[user_column].map(users_mapping.direct.get).values,
                dataframe[item_column].map(items_mapping.direct.get).values,
            ),
        )
    )


def get_objective_for_optuna(
    cv,
    interactions_df: pd.DataFrame,
    models_map: Dict,
    userknn_cls: Type,
    metrics_map: Dict,
    n_users: int = 10,
):
    """Get the objective for optuna tuning with cross-validation"""

    def objective(trial):
        fold_iterator = cv.split(interactions_df, collect_fold_stats=True)

        map10_buffer = []

        for i_fold, (train_ids, test_ids, fold_info) in enumerate(
            fold_iterator
        ):
            print(f"\n==================== Fold {i_fold}")
            print(fold_info)

            df_train = interactions_df.df.iloc[train_ids].copy()
            df_test = interactions_df.df.iloc[test_ids].copy()

            catalog = df_train["item_id"].unique()

            n_user = trial.suggest_int("n_users", 10, 50)
            model_name = trial.suggest_categorical(
                "model_name", ["cosine", "bm25", "tfidf"]
            )
            ModelType = models_map[model_name]

            userknn_model = userknn_cls(
                model=ModelType(K=n_user), N_users=n_users
            )
            userknn_model.fit(df_train)

            recos = userknn_model.predict(df_test)

            metric_values = calc_metrics(
                metrics_map,
                reco=recos,
                interactions=df_test,
                prev_interactions=df_train,
                catalog=catalog,
            )

            fold = {"fold": i_fold, "model": model_name, "n_users": n_user}
            fold.update(metric_values)

            map10_buffer.append(metric_values["map@10"])

        return np.array(map10_buffer).mean()

    return objective
