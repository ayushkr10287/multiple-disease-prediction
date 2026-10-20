import logging
import yaml
from pathlib import Path

import pandas as pd
from joblib import dump

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score,
)

from src.training.config.settings import Settings


def train_model():
    try:
        
        settings = Settings()

        DATASET_PATH = Path(settings.heart_disease_dataset_path)
        MODEL_PATH = Path(settings.heart_disease_model_path)
        LOG_PATH = Path(settings.log_path)
        HYPER_PARAMS_YAML_PATH = Path(settings.hyper_params_yaml_path)

        TARGET_COL = settings.heart_disease_target_col
        TEST_SIZE = settings.test_size
        RANDOM_STATE = settings.random_state

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        # Load data
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape: {df.shape}")

        # Separate X and y
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # Create a signature for each feature row to prevent duplicate leakage to test set
        row_signature = pd.util.hash_pandas_object(X, index=False)

        # Group-based split (same logic as notebook)
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(gss.split(X, y, groups=row_signature))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        with open(HYPER_PARAMS_YAML_PATH, "r") as file:
            hyperparams = yaml.safe_load(file)

        model_params = hyperparams["heart_disease"]["params"]

        best_rf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **model_params
        )

        # Keep scaler in pipeline to match notebook structure
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", best_rf)
            ]
        )

        pipeline.fit(X_train, y_train)
        logging.info("Model training completed")

        # Evaluation
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logging.info(f"Train Accuracy: {train_acc:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        logging.info(f"Test  Accuracy: {test_acc:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

        logging.info("Train Classification Report:\n" + classification_report(y_train, y_train_pred))
        logging.info("Test Classification Report:\n" + classification_report(y_test, y_test_pred))


        # save trained model
        dump(pipeline, MODEL_PATH)
        logging.info(f"Model saved to: {MODEL_PATH}")

        logging.info("Training Script Completed")


    except Exception as e:
        print(f"Training failed: {e}")
        logging.exception(f"Training Script Failed: {e}")
        raise


if __name__ == "__main__":
    train_model()
