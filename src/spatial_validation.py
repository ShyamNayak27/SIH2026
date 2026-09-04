"""
Spatial Validation Module for Spatial Landslide Susceptibility Baseline

Provides spatial block splitting and cross-validation utilities to evaluate
model generalization across geographically separated 25 km blocks (block_id).
Includes validation split integrity verification to prevent spatial leakage.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

SEED = 42


def get_spatial_splits(df, n_splits=5, strategy="spatial"):
    """
    Generates cross-validation fold indices.
    strategy='spatial': GroupKFold by block_id (spatially disjoint).
    strategy='random': StratifiedKFold (random sample split).
    """
    X = df.index.values
    y = df["label"].values
    groups = df["block_id"].values
    
    if strategy == "spatial":
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups=groups))
    elif strategy == "random":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y))
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")
        
    return splits


def verify_split_integrity(df, split_col="split"):
    """
    Verifies that spatial blocks do not cross train/val/test split boundaries,
    and checks for duplicate coordinates across splits.
    """
    if split_col not in df.columns:
        print(f"[WARN] Split column '{split_col}' not found in DataFrame.")
        return True
        
    # Check block disjointness
    block_split_counts = df.groupby("block_id")[split_col].nunique()
    leaking_blocks = (block_split_counts > 1).sum()
    
    # Check coordinate leakage
    df["coord_key"] = df["lon"].round(5).astype(str) + "_" + df["lat"].round(5).astype(str)
    coord_split_counts = df.groupby("coord_key")[split_col].nunique()
    leaking_coords = (coord_split_counts > 1).sum()
    
    print(f"--- Split Integrity Report ({split_col}) ---")
    print(f"Total rows: {len(df)}")
    print(f"Unique 25 km spatial blocks: {df['block_id'].nunique()}")
    print(f"Blocks crossing multiple splits: {leaking_blocks}")
    print(f"Coordinates crossing multiple splits: {leaking_coords}")
    
    assert leaking_blocks == 0, f"LEAKAGE DETECTED: {leaking_blocks} blocks cross splits!"
    assert leaking_coords == 0, f"LEAKAGE DETECTED: {leaking_coords} coordinates cross splits!"
    print("Split integrity verified successfully. Zero spatial block leakage.")
    return True


def get_canonical_train_val_test(df):
    """
    Extracts explicit Train, Validation, and Test sets based on pre-defined
    'split' column in ner_landslide_v1.csv (70% train, 15% val, 15% test).
    """
    verify_split_integrity(df, "split")
    
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    
    print(f"Canonical Train: {len(train_df)} rows ({int(train_df.label.sum())} pos, base rate = {train_df.label.mean():.3f})")
    print(f"Canonical Val:   {len(val_df)} rows ({int(val_df.label.sum())} pos, base rate = {val_df.label.mean():.3f})")
    print(f"Canonical Test:  {len(test_df)} rows ({int(test_df.label.sum())} pos, base rate = {test_df.label.mean():.3f})")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    from preprocessing import load_dataset
    df = load_dataset()
    verify_split_integrity(df)
    train_df, val_df, test_df = get_canonical_train_val_test(df)
