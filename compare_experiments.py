import mlflow
import pandas as pd


def get_best_experiment(
    experiment_name: str = "heart_disease_classification",
    metric_name: str = "metrics.f1_score",
):
    """Queries MLflow runs programmatically to identify the top performing run."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{metric_name} DESC"],
    )

    if runs.empty:
        print("No runs found for this experiment.")
        return

    print("\n=== All Experiment Runs ===")
    display_cols = [
        "run_id",
        "params.type",
        "metrics.f1_score",
        "metrics.accuracy",
        "metrics.roc_auc",
        "tags.data_version",
    ]
    existing_cols = [c for c in display_cols if c in runs.columns]
    print(runs[existing_cols].to_string(index=False))

    best_run = runs.iloc[0]
    print("\n🏆 BEST MODEL RUN:")
    print(f"Run ID: {best_run['run_id']}")
    print(f"Model Type: {best_run.get('params.type', 'N/A')}")
    print(f"Top F1 Score: {best_run['metrics.f1_score']:.4f}")
    print(f"Accuracy: {best_run['metrics.accuracy']:.4f}")

    return best_run


if __name__ == "__main__":
    get_best_experiment()
