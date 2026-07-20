import os
import sys
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def generate_simulated_production_data(
    df: pd.DataFrame, drift: bool = True
) -> pd.DataFrame:
    """Generates simulated production dataset with optional synthetic feature drift."""
    prod_df = df.copy()

    if drift:
        # Simulate drift in continuous features (e.g., patient age & cholesterol shifts)
        if "chol" in prod_df.columns:
            prod_df["chol"] = prod_df["chol"] * 1.35 + np.random.normal(
                10, 5, size=len(prod_df)
            )
        if "age" in prod_df.columns:
            prod_df["age"] = prod_df["age"] + np.random.normal(
                8, 2, size=len(prod_df)
            )

    return prod_df


def run_drift_monitoring(
    drift_threshold: float = 0.3, report_out_path: str = "reports/drift_report.html"
):
    """Runs Evidently data drift detection and exports an HTML report."""
    data_path = "data/heart_disease.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}")

    ref_data = pd.read_csv(data_path)
    prod_data = generate_simulated_production_data(ref_data, drift=True)

    os.makedirs(os.path.dirname(report_out_path), exist_ok=True)

    # Define Column Mapping
    column_mapping = ColumnMapping()
    column_mapping.target = "num"
    column_mapping.numerical_features = [
        c for c in ref_data.select_dtypes(include=[np.number]).columns if c != "num"
    ]

    # Generate Evidently Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=ref_data,
        current_data=prod_data,
        column_mapping=column_mapping,
    )

    # Save HTML Report
    drift_report.save_html(report_out_path)
    print(f"Evidently HTML drift report generated at: {report_out_path}")

    # Parse summary results
    report_dict = drift_report.as_dict()
    drift_metrics = report_dict["metrics"][0]["result"]
    number_of_drifted_features = drift_metrics["number_of_drifted_columns"]
    total_features = drift_metrics["number_of_columns"]
    drift_share = drift_metrics["share_of_drifted_columns"]

    print(f"\n--- DRIFT MONITORING SUMMARY ---")
    print(f"Total Features Monitored: {total_features}")
    print(f"Drifted Features Count  : {number_of_drifted_features}")
    print(
        f"Overall Drift Share     : {drift_share * 100:.2f}% (Threshold:"
        f" {drift_threshold * 100:.1f}%)"
    )

    if drift_share > drift_threshold:
        print("\n🚨 ALERT: Data drift threshold breached!")
        sys.exit(1)
    else:
        print("\n✅ Drift levels are within acceptable limits.")
        sys.exit(0)


if __name__ == "__main__":
    run_drift_monitoring()
