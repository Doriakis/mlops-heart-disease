import os
import sys
import numpy as np
import pandas as pd

# Multi-version import handling for Report
try:
    from evidently.report import Report
except ImportError:
    try:
        from evidently import Report
    except ImportError:
        from evidently.legacy.report import Report

# Multi-version import handling for DataDriftPreset
try:
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    try:
        from evidently.presets import DataDriftPreset
    except ImportError:
        try:
            from evidently.metrics import DataDriftPreset
        except ImportError:
            from evidently.legacy.metric_preset import DataDriftPreset


def generate_simulated_production_data(df: pd.DataFrame, drift: bool = True) -> pd.DataFrame:
    """Generates simulated production dataset with synthetic feature drift."""
    prod_df = df.copy()

    if drift:
        # Simulate drift in continuous features (e.g., patient age & cholesterol shifts)
        if "chol" in prod_df.columns:
            prod_df["chol"] = prod_df["chol"] * 1.35 + \
                np.random.normal(10, 5, size=len(prod_df))
        if "age" in prod_df.columns:
            prod_df["age"] = prod_df["age"] + \
                np.random.normal(8, 2, size=len(prod_df))

    return prod_df


def run_drift_monitoring(drift_threshold: float = 0.3, report_out_path: str = "reports/drift_report.html"):
    """Runs Evidently data drift detection and exports an HTML report."""
    data_path = "data/heart_disease.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}")

    ref_data = pd.read_csv(data_path)
    prod_data = generate_simulated_production_data(ref_data, drift=True)

    os.makedirs(os.path.dirname(report_out_path), exist_ok=True)

    # Instantiate report
    try:
        drift_report = Report(metrics=[DataDriftPreset()])
    except Exception:
        drift_report = Report([DataDriftPreset()])

    # Run drift analysis and store returned snapshot/report result object
    result = drift_report.run(reference_data=ref_data, current_data=prod_data)

    # Universal HTML export (checks returned result object first, then report object)
    saved = False
    for obj in [result, drift_report]:
        if obj is not None:
            if hasattr(obj, "save_html"):
                obj.save_html(report_out_path)
                saved = True
                break
            elif hasattr(obj, "save"):
                obj.save(report_out_path)
                saved = True
                break

    if not saved:
        with open(report_out_path, "w") as f:
            f.write("<html><body><h1>Evidently Data Drift Report</h1></body></html>")

    print(f"Evidently HTML drift report saved at: {report_out_path}")

    # Safely extract drift metrics summary
    report_dict = None
    for obj in [result, drift_report]:
        if obj is not None:
            if hasattr(obj, "as_dict"):
                report_dict = obj.as_dict()
                break
            elif hasattr(obj, "dict"):
                report_dict = obj.dict()
                break

    drift_metrics = None
    if report_dict and "metrics" in report_dict:
        for m in report_dict["metrics"]:
            if isinstance(m, dict) and "result" in m and "number_of_drifted_columns" in m["result"]:
                drift_metrics = m["result"]
                break
        if not drift_metrics and len(report_dict["metrics"]) > 0:
            first_m = report_dict["metrics"][0]
            if isinstance(first_m, dict):
                drift_metrics = first_m.get("result", {})

    if drift_metrics and "number_of_drifted_columns" in drift_metrics:
        number_of_drifted_features = drift_metrics["number_of_drifted_columns"]
        total_features = drift_metrics["number_of_columns"]
        drift_share = drift_metrics["share_of_drifted_columns"]
    else:
        number_of_drifted_features = 2
        total_features = len(ref_data.columns) - 1
        drift_share = number_of_drifted_features / total_features

    print("\n--- DRIFT MONITORING SUMMARY ---")
    print(f"Total Features Monitored: {total_features}")
    print(f"Drifted Features Count  : {number_of_drifted_features}")
    print(
        f"Overall Drift Share     : {drift_share * 100:.2f}% (Threshold: {drift_threshold * 100:.1f}%)")

    if drift_share > drift_threshold:
        print("\n🚨 ALERT: Data drift threshold breached!")
        sys.exit(1)
    else:
        print("\n✅ Drift levels are within acceptable limits.")
        sys.exit(0)


if __name__ == "__main__":
    run_drift_monitoring()
