# Adapted from https://github.com/snorkel-team/snorkel-tutorials/tree/master/visual_relation

import json
import os
import random
import subprocess

import numpy as np
import pandas as pd


def flatten_vr(img, relationship, objects, predicates, classify=None):
    """Create a per-relationship entry from a per-image entry JSON."""

    new_relationship_dict = {}
    new_relationship_dict["subject_category"] = objects[
        relationship["subject"]["category"]
    ]
    new_relationship_dict["object_category"] = objects[
        relationship["object"]["category"]
    ]
    new_relationship_dict["subject_bbox"] = tuple(relationship["subject"]["bbox"])
    new_relationship_dict["object_bbox"] = tuple(relationship["object"]["bbox"])

    new_relationship_dict["source_img"] = img

    if classify is None:
        new_relationship_dict["y"] = predicates[relationship["predicate"]]
    else:
        if predicates[relationship["predicate"]] in classify:
            new_relationship_dict["y"] = 1
        else:
            new_relationship_dict["y"] = 0

    return new_relationship_dict


def vr_to_pandas(
    relationships_set, objects, predicates, list_of_predicates, classify=None, keys_list=None
):
    """Create Pandas DataFrame from JSON of relationships."""

    relationships = []

    for img in relationships_set:
        if (keys_list is None) or (img in keys_list):
            img_relationships = relationships_set[img]
            for relationship in img_relationships:
                predicate_idx = relationship["predicate"]
                if predicates[predicate_idx] in list_of_predicates:
                    relationships.append(
                        flatten_vr(img, relationship, objects, predicates, classify)
                    )
        else:
            continue
    return pd.DataFrame.from_dict(relationships)


def df_drop_duplicates(df):
    """Drop duplicates for object pairs with multiple predicate labels"""

    np.random.seed(456)
    return df.sample(frac=1).sort_values("y").drop_duplicates(subset=df.columns.difference(["y"]), ignore_index=True, keep="first").sort_index()


def balance_dataset(df):
    """Balance classes in dataset"""

    np.random.seed(456)
    df = df.sample(frac=1).groupby("y")
    return pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()))).reset_index(drop=True)


def load_vr_data(classify=None, include_predicates=None, path_prefix="", drop_duplicates=False, balance=False, validation=True):
    """Load Pandas DataFrame of visual relations"""

    relationships_train = json.load(open(path_prefix + "data/annotations/annotations_train.json"))
    relationships_test = json.load(open(path_prefix + "data/annotations/annotations_test.json"))

    objects = json.load(open(path_prefix + "data/annotations/objects.json"))
    predicates = json.load(open(path_prefix + "data/annotations/predicates.json"))

    if include_predicates is None:
        include_predicates = predicates

    if validation:
        np.random.seed(123)
        val_idx = list(np.random.choice(len(relationships_train), 1000, replace=False))

        relationships_val = {
            key: value
            for i, (key, value) in enumerate(relationships_train.items())
            if i in val_idx
        }
        relationships_train = {
            key: value
            for i, (key, value) in enumerate(relationships_train.items())
            if i not in val_idx
        }

    train_df = vr_to_pandas(
        relationships_train,
        objects,
        predicates,
        list_of_predicates=include_predicates,
        classify=classify
    )

    test_df = vr_to_pandas(
        relationships_test,
        objects,
        predicates,
        list_of_predicates=include_predicates,
        classify=classify
    )

    if drop_duplicates:
        train_df = df_drop_duplicates(train_df)
        test_df = df_drop_duplicates(test_df)

    if balance:
        train_df = balance_dataset(train_df)

    if validation:
        train_df["labels"] = -1 * np.ones(len(train_df))
        valid_df = vr_to_pandas(
            relationships_val,
            objects,
            predicates,
            list_of_predicates=include_predicates,
            classify=classify
        )
        if drop_duplicates:
            valid_df = df_drop_duplicates(valid_df)

        return train_df, valid_df, test_df

    return train_df, test_df
