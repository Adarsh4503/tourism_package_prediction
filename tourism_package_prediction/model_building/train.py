"""Train, tune and log an XGBoost model for the tourism dataset."""
import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups",
    "DurationOfPitch",
]

CATEGORICAL_COLS = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
    "Designation", "ProductPitched",
]

DEPLOY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "deployment")
)
MODEL_FILENAME = "best_model.joblib"
EXPERIMENT_NAME = "tourism-wellness-prediction"


def _load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def _build_pipeline():
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_COLS),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        remainder="drop",
    )
    classifier = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    return make_pipeline(preprocessor, classifier)


def main() -> None:
    os.makedirs(DEPLOY_DIR, exist_ok=True)

    Xtrain, Xtest, ytrain, ytest = _load_splits()
    pipeline = _build_pipeline()

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.05, 0.1],
        "xgbclassifier__subsample": [0.8, 1.0],
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    grid.fit(Xtrain, ytrain)

    cv_results = pd.DataFrame(grid.cv_results_)

    with mlflow.start_run(run_name="tuning-parent") as parent_run:
        # Log every parameter combination that was evaluated.
        # sklearn prefixes these as `param_<name>` in cv_results_.
        for idx, row in cv_results.iterrows():
            params = {k: row[f"param_{k}"] for k in param_grid.keys()}
            with mlflow.start_run(
                run_name=f"trial-{idx}", nested=True
            ):
                mlflow.log_params(params)
                mlflow.log_metric("cv_mean_f1", float(row["mean_test_score"]))
                mlflow.log_metric("cv_std_f1", float(row["std_test_score"]))
                mlflow.log_metric("rank", int(row["rank_test_score"]))

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        # Final evaluation on the held-out test set.
        y_pred = best_model.predict(Xtest)
        accuracy = accuracy_score(ytest, y_pred)
        report = classification_report(ytest, y_pred, output_dict=True)

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_f1", best_cv_score)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision_weighted",
                         report["weighted avg"]["precision"])
        mlflow.log_metric("test_recall_weighted",
                         report["weighted avg"]["recall"])
        mlflow.log_metric("test_f1_weighted",
                         report["weighted avg"]["f1-score"])
        mlflow.set_tag("best_run_id", parent_run.info.run_id)

        model_path = os.path.join(DEPLOY_DIR, MODEL_FILENAME)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        print("Trials evaluated:", len(cv_results))
        print("Best params:", best_params)
        print(f"Best CV F1: {best_cv_score:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print("Classification report:")
        print(classification_report(ytest, y_pred))
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
