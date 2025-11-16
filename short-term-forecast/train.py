import os
import pandas as pd
import joblib
from lightgbm import LGBMRegressor


def train_all_group_models(
    X_train_path: str = "../data/second/X_train.csv",
    y_train_path: str = "../data/second/y_train.csv",
    output_dir: str = "../data/second/models",
) -> None:
    """
    Train one LightGBM model per user group.

    Requirements:
    -----------
    X_train.csv columns:
        - group_id
        - lag_1, lag_24, lag_168
        - hour, day_of_week, is_weekend
        - price_t

    y_train.csv:
        - single column with target consumption values
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X_train = pd.read_csv(X_train_path)
    y_train_df = pd.read_csv(y_train_path)

    # Robustly get target as a Series (handles any column name)
    if y_train_df.shape[1] != 1:
        raise ValueError("y_train.csv must contain exactly one column (the target).")
    y_train = y_train_df.iloc[:, 0]

    # Basic checks
    if "group_id" not in X_train.columns:
        raise ValueError("X_train.csv must contain a 'group_id' column.")

    feature_cols = [
        "lag_1",
        "lag_24",
        "lag_168",
        "hour",
        "day_of_week",
        "is_weekend",
        "price_t",
    ]
    for col in feature_cols:
        if col not in X_train.columns:
            raise ValueError(f"Missing feature column in X_train: {col}")

    cat_features = ["hour", "day_of_week", "is_weekend"]

    groups = sorted(X_train["group_id"].unique())
    print(f"Found {len(groups)} user groups. Training one model per group.")

    for g in groups:
        print(f"\n=== Training model for group {g} ===")

        # Select this group's rows
        mask = X_train["group_id"] == g
        X_g = X_train.loc[mask, feature_cols].copy()
        y_g = y_train.loc[mask].copy()

        if len(X_g) == 0:
            print(f"Skipping group {g}: no training rows found.")
            continue

        # Define model (tune as needed)
        model = LGBMRegressor(
            objective="regression",
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
        )

        # Fit model
        model.fit(
            X_g,
            y_g,
            categorical_feature=cat_features,
        )

        # Save model
        model_path = os.path.join(output_dir, f"lgbm_group_{g}.pkl")
        joblib.dump(model, model_path)

        print(f"Saved model for group {g} -> {model_path}")

    print("\nAll available groups processed.")


if __name__ == "__main__":
    train_all_group_models()
