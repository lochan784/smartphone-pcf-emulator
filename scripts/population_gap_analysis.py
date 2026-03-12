import pandas as pd
import json
import sys

def analyze_population_gap(features_path, preds_path):
    """
    Identifies the exact discrepancy between the feature catalog and model predictions.
    """
    df_feat = pd.read_csv(features_path)
    df_pred = pd.read_csv(preds_path)

    # Standardize keys for comparison
    df_feat['model_key'] = df_feat['model'].astype(str).str.strip().str.lower()
    df_pred['model_key'] = df_pred['model'].astype(str).str.strip().str.lower()

    feat_set = set(df_feat['model_key'])
    pred_set = set(df_pred['model_key'])

    intersection = feat_set.intersection(pred_set)
    only_in_features = list(feat_set - pred_set)
    only_in_preds = list(pred_set - feat_set)

    # Brand-level analysis
    feat_brands = df_feat['brand'].unique().tolist() if 'brand' in df_feat.columns else "Not Found"
    
    report = {
        "summary": {
            "total_featured_devices": len(feat_set),
            "total_predicted_devices": len(pred_set),
            "overlap_count": len(intersection)
        },
        "brand_diagnostics": {
            "featured_brands": feat_brands,
            "prediction_brands": df_pred['brand'].unique().tolist() if 'brand' in df_pred.columns else "Unknown"
        },
        "mismatch_samples": {
            "missing_predictions_for": only_in_features[:5],
            "predictions_without_features": only_in_preds[:5]
        }
    }

    return report

if __name__ == "__main__":
    FEAT_FILE = "data/processed/smartphones_structured.csv"
    PRED_FILE = "data/processed/posterior_predictions_gold.csv"
    
    try:
        report = analyze_population_gap(FEAT_FILE, PRED_FILE)
        print(f"--- Population Gap Report ---\n{json.dumps(report, indent=4)}")
        
        if report["summary"]["overlap_count"] == 0:
            print("\nCRITICAL ACTION: Your model output contains 0% of your featured devices.")
            print("Check if '04_fit_bayesian_emulator.py' is limited to specific brands.")
    except Exception as e:
        print(f"Audit Failed: {str(e)}")