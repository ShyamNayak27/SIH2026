import pandas as pd

from config import MODEL_FEATURES


def load_dataset(filepath):

    df = pd.read_csv(filepath)

    return df


def get_model_data(df):

    X = df[MODEL_FEATURES]

    y = df["label"]

    return X, y


def get_train_test_data(df):

    train_df = df[df["split"] == "train"]

    test_df = df[df["split"] == "test"]

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df["label"]

    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["label"]

    return X_train, X_test, y_train, y_test