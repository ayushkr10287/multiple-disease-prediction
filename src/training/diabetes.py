import logging
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score

from src.common.preprocessing_util import replace_zeros_with_nan
from src.training.config.settings import Settings


def train_diabetes_model():
    try:
        settings = Settings()

        DATASET_PATH = Path(settings.diabetes_dataset_path)
        MODEL_PATH = Path(settings.diabetes_model_path)
        LOG_PATH = Path(settings.log_path)
        HYPER_PARAMS_YAML_PATH = Path(settings.hyper_params_yaml_path)

        TARGET_COL = settings.diabetes_target_col
        TEST_SIZE = settings.test_size
        RANDOM_STATE = settings.random_state

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH),
            ],
        )

        logging.info("Starting Diabetes Model Training")

        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape: {df.shape}")

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("zero_to_nan", FunctionTransformer(replace_zeros_with_nan, validate=False)),
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                )
            ],
            remainder="drop",
        )

        with open(HYPER_PARAMS_YAML_PATH, "r") as file:
            hyperparams = yaml.safe_load(file)

        model_params = hyperparams["diabetes"]["params"]

        model = SVC(
            random_state=RANDOM_STATE,
            **model_params,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        logging.info("Model training completed")

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        logging.info(
            f"Train Acc: {accuracy_score(y_train, y_train_pred):.4f} | "
            f"Recall: {recall_score(y_train, y_train_pred):.4f} | "
            f"F1: {f1_score(y_train, y_train_pred):.4f}"
        )

        logging.info(
            f"Test  Acc: {accuracy_score(y_test, y_test_pred):.4f} | "
            f"Recall: {recall_score(y_test, y_test_pred):.4f} | "
            f"F1: {f1_score(y_test, y_test_pred):.4f}"
        )

        logging.info("Train Classification Report:\n" + classification_report(y_train, y_train_pred))
        logging.info("Test Classification Report:\n" + classification_report(y_test, y_test_pred))

        dump(pipeline, MODEL_PATH)
        logging.info(f"Model saved at: {MODEL_PATH}")
        logging.info("Diabetes Training Completed")

    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    train_diabetes_model()
    