import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split


DATA_DIR = "."
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUBMISSION_PATH = f"{DATA_DIR}/Prediction by XGBoost_clean.csv"


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def drop_low_quality_columns(df, missing_ratio_threshold=0.8):
    cols_to_drop = [
        col for col in df.columns
        if col != "SalePrice" and df[col].isna().mean() > missing_ratio_threshold
    ]
    return df.drop(columns=cols_to_drop, errors="ignore")


def fill_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if col == "SalePrice":
            continue
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].mode().empty:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def preprocess_data(train_df, test_df):
    combined = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    if "Id" in combined.columns:
        ids = combined["Id"].copy()
        combined = combined.drop(columns=["Id"])
    else:
        ids = pd.Series(index=combined.index, dtype=int)

    combined = drop_low_quality_columns(combined)
    combined = fill_missing_values(combined)
    combined = pd.get_dummies(combined, drop_first=True)

    train_len = len(train_df)
    X_train = combined.iloc[:train_len].copy()
    X_test = combined.iloc[train_len:].copy()

    y_train = train_df["SalePrice"].reset_index(drop=True)

    if "SalePrice" in X_train.columns:
        X_train = X_train.drop(columns=["SalePrice"])

    if "SalePrice" in X_test.columns:
        X_test = X_test.drop(columns=["SalePrice"])

    return X_train, X_test, y_train, ids.iloc[train_len:]


def build_model():
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


def main():
    train_df, test_df = load_data()

    X_train, X_test, y_train, test_ids = preprocess_data(train_df, test_df)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
    )

    model = build_model()
    model.fit(X_tr, y_tr)

    val_predictions = model.predict(X_val)
    rmsle = np.sqrt(mean_squared_log_error(np.abs(y_val), np.abs(val_predictions)))
    print(f"Validation RMSLE: {rmsle:.4f}")

    final_predictions = model.predict(X_test)
    submission = pd.DataFrame({
        "Id": test_ids.reset_index(drop=True),
        "SalePrice": final_predictions,
    })
    print(submission.head())
    # submission.to_csv(SUBMISSION_PATH, index=False)
    # print(f"Submission saved to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
