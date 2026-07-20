import os
import sys
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import load_data, preprocess_data, split_data
from src.evaluate import evaluate_model


def get_dvc_hash(dvc_file_path: str = "data/heart_disease.csv.dvc") -> str:
    """Extracts md5 hash from .dvc tracking file if available."""
    if os.path.exists(dvc_file_path):
        with open(dvc_file_path, "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split(":")[1].strip()
    return "v1.0-raw"


def train_model(config_path: str = "configs/config.yaml", params_override=None):
    """Trains model according to config settings and logs run to MLflow."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if params_override:
        for key, val in params_override.get("model", {}).items():
            config["model"][key] = val

    # Load & Preprocess Data
    raw_df = load_data(config["data"]["raw_path"])
    X, y = preprocess_data(raw_df, target_col=config["data"]["target_column"])
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    # Instantiate Model Type
    model_type = config["model"]["type"]
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config["model"].get("n_estimators", 100),
            max_depth=config["model"].get("max_depth", 5),
            random_state=config["data"]["random_state"]
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            C=config["model"].get("C", 1.0),
            max_iter=config["model"].get("max_iter", 500),
            random_state=config["data"]["random_state"]
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=config["model"].get("n_estimators", 100),
            learning_rate=config["model"].get("learning_rate", 0.1),
            max_depth=config["model"].get("max_depth", 3),
            random_state=config["data"]["random_state"]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Configure MLflow
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    data_version = get_dvc_hash()

    with mlflow.start_run() as run:
        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log Hyperparameters & Tags
        mlflow.log_params(config["model"])
        mlflow.log_param("data_path", config["data"]["raw_path"])
        mlflow.set_tag("data_version", data_version)

        # Log Metrics
        mlflow.log_metrics(metrics)

        # Log Model Artifact
        mlflow.sklearn.log_model(model, "model")

        print(
            f"Run ID: {run.info.run_id} | Model: {model_type} | F1: {metrics['f1_score']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

        return run.info.run_id, metrics


if __name__ == "__main__":
    run_id, metrics = train_model()
    print(
        f"Training completed successfully! F1 Score: {metrics['f1_score']:.4f}")
    if metrics["f1_score"] < 0.75:
        print("🚨 Model failed performance threshold (F1 < 0.75)!")
        sys.exit(1)
    else:
        print("✅ Model passed performance threshold (F1 >= 0.75).")
        sys.exit(0)
