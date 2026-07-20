from src.evaluate import evaluate_model
from src.data_preprocessing import load_data, preprocess_data, split_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import mlflow.sklearn
import mlflow
import yaml
import os
import sys
import traceback
from pathlib import Path

# Ensure project root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_dvc_hash(dvc_file_path: str = "data/heart_disease.csv.dvc") -> str:
    """Extracts md5 hash from .dvc tracking file if available."""
    if os.path.exists(dvc_file_path):
        with open(dvc_file_path, "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split(":")[1].strip()
    return "v1.0-raw"


def ensure_config_exists(config_path: str = "configs/config.yaml") -> dict:
    """Ensures configuration file exists, creating a default one if missing."""
    default_config = {
        "data": {
            "raw_path": "data/heart_disease.csv",
            "target_column": "num",
            "test_size": 0.2,
            "random_state": 42
        },
        "model": {
            "type": "random_forest",
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        },
        "mlflow": {
            "experiment_name": "heart_disease_classification"
        }
    }

    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(default_config, f)
        return default_config

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(config_path: str = "configs/config.yaml", params_override=None):
    """Trains model according to config settings and logs run to MLflow."""
    config = ensure_config_exists(config_path)

    if params_override:
        for key, val in params_override.get("model", {}).items():
            config["model"][key] = val

    # Ensure dataset exists, running download_data if needed
    raw_path = config["data"]["raw_path"]
    if not os.path.exists(raw_path):
        print(
            f"Dataset missing at {raw_path}. Triggering download_data script...")
        from download_data import main as download_main
        download_main()

    # Load & Preprocess Data
    raw_df = load_data(raw_path)
    X, y = preprocess_data(raw_df, target_col=config["data"]["target_column"])
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    # Instantiate Model
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
    try:
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
    except Exception as e:
        print(f"MLflow set_experiment warning: {e}")

    data_version = get_dvc_hash()

    with mlflow.start_run() as run:
        # Train & Evaluate
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Log to MLflow with error protection
        try:
            mlflow.log_params(config["model"])
            mlflow.log_param("data_path", raw_path)
            mlflow.set_tag("data_version", data_version)
            mlflow.log_metrics(metrics)
        except Exception as e:
            print(f"MLflow logging params/metrics warning: {e}")

        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                print(f"MLflow model artifact logging warning: {e}")

        print(
            f"Run ID: {run.info.run_id} | Model: {model_type} | F1: {metrics['f1_score']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

        return run.info.run_id, metrics


if __name__ == "__main__":
    try:
        run_id, metrics = train_model()
        print(
            f"Training completed successfully! F1 Score: {metrics['f1_score']:.4f}")
        if metrics["f1_score"] < 0.75:
            print("🚨 Model failed performance threshold (F1 < 0.75)!")
            sys.exit(1)
        else:
            print("✅ Model passed performance threshold (F1 >= 0.75).")
            sys.exit(0)
    except Exception as e:
        print(f"❌ Error during training execution: {e}")
        traceback.print_exc()
        sys.exit(1)
