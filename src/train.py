"""
Model Training & Hyperparameter Tuning Module

Trains mandatory baseline models (Dummy, Logistic Regression, Random Forest, 
Gradient Boosting) using leakage-safe pipelines and spatial block cross-validation.
"""
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from preprocessing import load_dataset, get_feature_sets, create_preprocessing_pipeline
from spatial_validation import get_canonical_train_val_test

SEED = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def get_model_factories(scale_pos_weight=2.0):
    """
    Returns dictionaries of un-tuned model constructors and hyperparameter search grids.
    """
    models = {
        "Dummy": lambda: DummyClassifier(strategy="prior"),
        "LogisticRegression": lambda: LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED),
        "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED)
    }
    
    param_grids = {
        "LogisticRegression": {
            "model__C": np.logspace(-3, 2, 20),
            "model__penalty": ["l2"]
        },
        "RandomForest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [4, 6, 8, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "GradientBoosting": {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0]
        }
    }
    
    return models, param_grids


def build_pipeline(model_instance, feature_cols, scale_numeric=True):
    """
    Assembles scikit-learn Pipeline combining preprocessor and model.
    """
    preprocessor = create_preprocessing_pipeline(feature_cols, scale_numeric=scale_numeric)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_instance)
    ])
    return pipeline


def train_model(model_name, model_instance, train_df, feature_cols, scale_numeric=True):
    """
    Trains a model pipeline on training data.
    """
    pipeline = build_pipeline(model_instance, feature_cols, scale_numeric=scale_numeric)
    X_tr = train_df[feature_cols]
    y_tr = train_df["label"].values
    
    pipeline.fit(X_tr, y_tr)
    return pipeline


def tune_hyperparameters(model_name, model_instance, param_grid, train_df, feature_cols, n_iter=10):
    """
    Performs RandomizedSearchCV using spatial GroupKFold cross-validation on block_id.
    """
    pipeline = build_pipeline(model_instance, feature_cols)
    X_tr = train_df[feature_cols]
    y_tr = train_df["label"].values
    groups = train_df["block_id"].values
    
    cv = GroupKFold(n_splits=5)
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=SEED,
        n_jobs=-1
    )
    
    search.fit(X_tr, y_tr, groups=groups)
    print(f"[{model_name}] Best Spatial CV ROC-AUC: {search.best_score_:.3f}")
    print(f"[{model_name}] Best Params: {search.best_params_}")
    return search.best_estimator_


if __name__ == "__main__":
    df = load_dataset()
    train_df, val_df, test_df = get_canonical_train_val_test(df)
    features, exp_name = get_feature_sets(df, "A")
    
    models, grids = get_model_factories()
    trained_pipelines = {}
    
    for name, factory in models.items():
        print(f"Training {name} ...")
        pipe = train_model(name, factory(), train_df, features)
        trained_pipelines[name] = pipe
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{name.lower()}_model.pkl"))
    print("Training complete.")
