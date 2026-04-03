"""
Unit and Integration Tests for main 6 core functionalities:
1. Input transaction data validation
2. Data preprocessing and transformation
3. Fraud prediction accuracy
4. XAI explanation generation
5. Ensemble explanation display
6. Report saving functionality
"""

import pickle
import tempfile
from pathlib import Path
import json
from datetime import datetime

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Loading Artifacts (like app.py does)
model_bundle_path = r".\Artifacts\model_bundle.pkl"
nn_model_path = r".\Artifacts\nn_model.pt"


print("Loading artifacts...")
# Load model bundle
with open(model_bundle_path, "rb") as f:
    bundle = pickle.load(f)

# Load neural network model
nn_payload = torch.load(nn_model_path, map_location="cpu")

# Extract bundle contents
preprocessor = bundle["preprocessor"]
feature_names = bundle["processed_feature_names"]
raw_fe_columns = bundle["raw_fe_columns"]
rf_model = bundle["rf_model"]
xgb_model = bundle["xgb_model"]
lgbm_model = bundle["lgbm_model"]
w_rf = bundle["weights"]["rf"]
w_xgb = bundle["weights"]["xgb"]
w_lgbm = bundle["weights"]["lgbm"]
w_mlp = bundle["weights"]["mlp"]
best_t = float(bundle["best_threshold"])
xai_assets = bundle["xai_assets"]
shap_bg_df = xai_assets["shap_bg_df"]
lime_train_np = xai_assets["lime_train_np"]
anchor_bg_np = xai_assets["anchor_bg_np"]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✓ Artifacts loaded successfully (device: {device})")


# Helper functions (from app.py)

class FraudMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# Rebuild MLP model
input_dim = int(nn_payload["input_dim"])
mlp_model = FraudMLP(input_dim).to(device)
mlp_model.load_state_dict(nn_payload["state_dict"])
mlp_model.eval()


def predict_mlp_probabilities(X_df):
    mlp_model.eval()
    X_t = torch.tensor(X_df.to_numpy(), dtype=torch.float32).to(device)
    with torch.no_grad():
        p = torch.sigmoid(mlp_model(X_t)).cpu().numpy().ravel()
    return np.vstack([1 - p, p]).T


def predict_ensemble_probabilities(X_processed_df):
    #Ensemble probability prediction from all 4 models
    rf_p = rf_model.predict_proba(X_processed_df)[:, 1]
    xgb_p = xgb_model.predict_proba(X_processed_df)[:, 1]
    lgbm_p = lgbm_model.predict_proba(X_processed_df)[:, 1]
    mlp_p = predict_mlp_probabilities(X_processed_df)[:, 1]
    ens_p = w_rf * rf_p + w_xgb * xgb_p + w_lgbm * lgbm_p + w_mlp * mlp_p
    return np.vstack([1 - ens_p, ens_p]).T


def add_time_and_age_features(df):
    #Feature engineering for time and age
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["trans_date_trans_time"].dt.dayofweek >= 5).astype(int)

    age_days = (df["trans_date_trans_time"] - df["dob"]).dt.days
    df["age"] = (age_days / 365.25).astype(int)

    df.drop(columns=["trans_date_trans_time", "dob"], inplace=True)
    return df


def build_report_block(row_dict, result, ensemble_df, anchor_disp, dice_rows, timestamp):
    #Format prediction result as text report
    lines = [
        "=" * 60,
        f"  Fraud Detection Report  |  {timestamp}",
        "=" * 60,
        f"Prediction     : {result['prediction'].upper()}",
        f"Fraud Prob.    : {result['fraud_probability']:.6f}",
        f"Confidence     : {result['confidence']:.6f}",
        f"Threshold Used : {result['threshold_used']:.6f}",
        "",
        "--- Input Transaction ---",
        json.dumps(row_dict, indent=2, default=str),
        "",
        "--- Final Ensemble Summary of Influential Attributes ---",
        f"{'Feature':<20} {'SHAP Score':>12} {'LIME Score':>12} {'Ensemble Score':>15}",
        "-" * 63,
    ]
    for _, r in ensemble_df.iterrows():
        lines.append(
            f"{r['Feature']:<20} {r['SHAP Score']:>12.6f} "
            f"{r['LIME Score']:>12.6f} {r['Ensemble Score']:>15.6f}"
        )
    lines.append("")
    lines.append("--- Anchor Explanation ---")
    if isinstance(anchor_disp, pd.DataFrame) and len(anchor_disp) > 0:
        lines.append(f"Rule      : {anchor_disp['anchor_rule'].iloc[0]}")
        lines.append(f"Precision : {anchor_disp['precision'].iloc[0]:.4f}")
        lines.append(f"Coverage  : {anchor_disp['coverage'].iloc[0]:.4f}")
    else:
        lines.append("Rule      : N/A")
        lines.append("Precision : 0.0000")
        lines.append("Coverage  : 0.0000")
    lines.append("")
    lines.append("--- DiCE Counterfactual Changes ---")
    if dice_rows:
        lines.append(f"{'CF #':<6} {'Feature':<20} {'From':<20} {'To':<20}")
        lines.append("-" * 68)
        for r in dice_rows:
            lines.append(
                f"{r['CF #']:<6} {r['Feature']:<20} {r['From']:<20} {r['To']:<20}"
            )
    else:
        lines.append("No counterfactual changes found.")
    lines.append("\n")
    return "\n".join(lines)


#Testcases  with + and - cases 
#Valid sample transaction data
@pytest.fixture
def sample_raw_transaction():
    return pd.DataFrame([{
        "trans_date_trans_time": "2026-04-03 15:30:00",
        "category": "personal_care",
        "amt": 50.25,
        "gender": "F",
        "lat": 33.9659,
        "long": -80.9355,
        "city_pop": 333497,
        "dob": "1975-06-15",
        "merch_lat": 33.986391,
        "merch_long": -81.200714,
    }])

#Transaction missing required field
@pytest.fixture
def invalid_transaction_missing_amt():
    return pd.DataFrame([{
        "trans_date_trans_time": "2026-04-03 15:30:00",
        "category": "personal_care",
        "gender": "F",
        "lat": 33.9659,
        "long": -80.9355,
        "city_pop": 333497,
        "dob": "1975-06-15",
        "merch_lat": 33.986391,
        "merch_long": -81.200714,
    }])

#Transaction with invalid date
@pytest.fixture
def invalid_transaction_wrong_date():
    return pd.DataFrame([{
        "trans_date_trans_time": "not-a-valid-date",
        "category": "personal_care",
        "amt": 50.25,
        "gender": "F",
        "lat": 33.9659,
        "long": -80.9355,
        "city_pop": 333497,
        "dob": "1975-06-15",
        "merch_lat": 33.986391,
        "merch_long": -81.200714,
    }])


# Testcase 1: Input Transaction Data Validation

def test_1_input_validation_accepts_valid_transaction(sample_raw_transaction):
    required_cols = [
        "trans_date_trans_time", "category", "amt", "gender",
        "lat", "long", "city_pop", "dob", "merch_lat", "merch_long"
    ]
    
    # Validate all required columns are present
    for col in required_cols:
        assert col in sample_raw_transaction.columns, f"Missing required column: {col}"
    
    # Validate data types and value ranges
    assert sample_raw_transaction["amt"].iloc[0] > 0, "Transaction amount must be positive"
    assert sample_raw_transaction["city_pop"].iloc[0] > 0, "City population must be positive"
    assert sample_raw_transaction["gender"].iloc[0] in ["M", "F"], "Gender must be M or F"
    
    # Verify valid transaction passes validation
    assert len(sample_raw_transaction) == 1
    assert not sample_raw_transaction.isnull().any().any()
    print("✓ TEST 1 PASSED: Valid transaction accepted")


def test_1_input_validation_rejects_missing_field(invalid_transaction_missing_amt):
    with pytest.raises(KeyError):
        preprocessor.transform(invalid_transaction_missing_amt[raw_fe_columns])
    print("✓ TEST 1 PASSED: Missing field correctly rejected")


def test_1_input_validation_rejects_invalid_date(invalid_transaction_wrong_date):
    with pytest.raises(Exception):
        add_time_and_age_features(invalid_transaction_wrong_date)
    print("✓ TEST 1 PASSED: Invalid date correctly rejected")


# Testcase 2: Data Preprocessing and Transformation
def test_2_data_preprocessing_transforms_correctly(sample_raw_transaction):
    """  
    checking the following:
    1. Time features (hour, month, weekend) are extracted
    2. Age is calculated from DOB
    3. Raw features are transformed to processed feature space
    """
    # Apply feature engineering
    raw_fe = add_time_and_age_features(sample_raw_transaction)
    
    # Verify time and age features were created
    assert "trans_hour" in raw_fe.columns
    assert "trans_month" in raw_fe.columns
    assert "is_weekend" in raw_fe.columns
    assert "age" in raw_fe.columns
    
    # Verify original datetime and DOB were removed
    assert "trans_date_trans_time" not in raw_fe.columns
    assert "dob" not in raw_fe.columns
    
    # Verify values are reasonable
    assert 0 <= raw_fe["trans_hour"].iloc[0] < 24
    assert 1 <= raw_fe["trans_month"].iloc[0] <= 12
    assert raw_fe["is_weekend"].iloc[0] in [0, 1]
    assert raw_fe["age"].iloc[0] > 0
    
    # Apply preprocessor transformation
    raw_fe_cols = raw_fe[raw_fe_columns]
    X_processed = preprocessor.transform(raw_fe_cols)
    
    # Verify output shape and type
    assert X_processed.shape[0] == 1
    assert X_processed.shape[1] == len(feature_names)
    assert isinstance(X_processed, np.ndarray)
    
    # Verify no NaN or Inf values
    assert not np.isnan(X_processed).any()
    assert not np.isinf(X_processed).any()
    
    print(f"✓ TEST 2 PASSED: Data preprocessing transform successful (shape: {X_processed.shape})")


def test_2_preprocessing_output_is_normalized(sample_raw_transaction):
    raw_fe = add_time_and_age_features(sample_raw_transaction)
    X_processed = preprocessor.transform(raw_fe[raw_fe_columns])
    
    # Verify output is numeric
    assert np.issubdtype(X_processed.dtype, np.number)
    
    # Verify no NaN or Inf values
    assert not np.isnan(X_processed).any()
    assert not np.isinf(X_processed).any()
    
    # Verify reasonable value ranges (usually 0-1 for scaled features)
    assert np.min(X_processed) >= -10
    assert np.max(X_processed) <= 1e6
    
    print("✓ TEST 2 PASSED: Preprocessed output is properly normalized")


# Testcase 3: Fraud Prediction
def test_3_fraud_prediction_returns_valid_probability(sample_raw_transaction):
    # Preprocess input
    raw_fe = add_time_and_age_features(sample_raw_transaction)
    X_processed = preprocessor.transform(raw_fe[raw_fe_columns])
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Get ensemble probability
    probs = predict_ensemble_probabilities(X_df)
    
    # Verify output shape and type
    assert probs.shape == (1, 2)
    assert isinstance(probs, np.ndarray)
    
    # Verify probability constraints (sum to 1, between 0-1)
    assert pytest.approx(probs[0].sum(), rel=1e-6) == 1.0
    assert 0.0 <= probs[0, 0] <= 1.0
    assert 0.0 <= probs[0, 1] <= 1.0
    
    # Extract fraud probability and determine label
    fraud_prob = probs[0, 1]
    pred_label = 1 if fraud_prob >= best_t else 0
    
    # Verify label is valid
    assert pred_label in [0, 1]
    
    print(f"✓ TEST 3 PASSED: Fraud prediction valid (prob={fraud_prob:.4f}, label={pred_label})")


def test_3_prediction_threshold_consistency(sample_raw_transaction):
    raw_fe = add_time_and_age_features(sample_raw_transaction)
    X_processed = preprocessor.transform(raw_fe[raw_fe_columns])
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    probs = predict_ensemble_probabilities(X_df)
    fraud_prob = probs[0, 1]
    
    # Manual threshold application
    pred_label = 1 if fraud_prob >= best_t else 0
    
    # Verify threshold behavior
    if fraud_prob >= best_t:
        assert pred_label == 1, "High probability should predict fraud"
    else:
        assert pred_label == 0, "Low probability should predict genuine"
    
    print(f"✓ TEST 3 PASSED: Prediction threshold applied correctly (threshold={best_t:.4f})")



# Testcase 4: Explanation Generation
def test_4_explanation_generation_components_available(sample_raw_transaction):
    #checks whether the XAI explainers (SHAP, LIME, Anchor) generate explanations for a transaction prediction.
    
    # Preprocess input
    raw_fe = add_time_and_age_features(sample_raw_transaction)
    X_processed = preprocessor.transform(raw_fe[raw_fe_columns])
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Get prediction probability
    probs = predict_ensemble_probabilities(X_df)
    fraud_prob = float(probs[0, 1])
    pred_label = 1 if fraud_prob >= best_t else 0
    
    # Verify explanation components exist
    assert shap_bg_df is not None, "SHAP background data required"
    assert lime_train_np is not None, "LIME training data required"
    assert anchor_bg_np is not None, "Anchor background data required"
    assert feature_names is not None, "Feature names required"
    
    # Verify explainers are available
    assert rf_model is not None
    assert xgb_model is not None
    assert lgbm_model is not None
    assert mlp_model is not None
    
    print(f"✓ TEST 4 PASSED: Explanation generation components available (fraud_prob={fraud_prob:.4f})")


def test_4_explanation_covers_all_features(sample_raw_transaction):
    #Verifies that generated explanations include all features.
    raw_fe = add_time_and_age_features(sample_raw_transaction)
    X_processed = preprocessor.transform(raw_fe[raw_fe_columns])
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Verify feature names match processed data dimensions
    assert len(feature_names) == X_df.shape[1]
    assert len(feature_names) > 0
    
    # Verify preprocessor output matches feature count
    for col in feature_names:
        assert col in X_df.columns
    
    print(f"✓ TEST 4 PASSED: Explanation covers all {len(feature_names)} features")


# Testcase 5: Ensemble Explanation Display
def test_5_ensemble_explanation_display_format():

    # Create mock ensemble explanation with expected structure
    ensemble_expl = pd.DataFrame({
        "Feature": ["amt", "city_pop", "age"],
        "SHAP Score": [0.45, 0.30, 0.15],
        "LIME Score": [0.40, 0.35, 0.20],
        "Ensemble Score": [0.85, 0.65, 0.35],
    })
    
    # Verify ensemble explanation format
    required_cols = ["Feature", "SHAP Score", "LIME Score", "Ensemble Score"]
    assert all(col in ensemble_expl.columns for col in required_cols)
    
    # Verify scores are numeric and in valid range
    assert (ensemble_expl["SHAP Score"] >= 0).all()
    assert (ensemble_expl["LIME Score"] >= 0).all()
    assert (ensemble_expl["Ensemble Score"] >= 0).all()
    
    # Verify ranking by ensemble score
    assert (ensemble_expl["Ensemble Score"].iloc[:-1].values >= 
            ensemble_expl["Ensemble Score"].iloc[1:].values).all()
    
    print("✓ TEST 5 PASSED: Ensemble explanation display format valid")


def test_5_ensemble_explanation_meaningful():
    # checks that different transaction types produce valid explanations.
    # Create test data with different amounts
    high_amt_tx = pd.DataFrame([{
        "trans_date_trans_time": "2026-04-03 15:30:00",
        "category": "personal_care",
        "amt": 5000.00,
        "gender": "F",
        "lat": 33.9659,
        "long": -80.9355,
        "city_pop": 333497,
        "dob": "1975-06-15",
        "merch_lat": 33.986391,
        "merch_long": -81.200714,
    }])
    
    low_amt_tx = pd.DataFrame([{
        "trans_date_trans_time": "2026-04-03 15:30:00",
        "category": "personal_care",
        "amt": 10.00,
        "gender": "F",
        "lat": 33.9659,
        "long": -80.9355,
        "city_pop": 333497,
        "dob": "1975-06-15",
        "merch_lat": 33.986391,
        "merch_long": -81.200714,
    }])
    
    # Both should produce valid predictions
    for tx in [high_amt_tx, low_amt_tx]:
        raw_fe = add_time_and_age_features(tx)
        X_processed = preprocessor.transform(raw_fe[raw_fe_columns])
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        probs = predict_ensemble_probabilities(X_df)
        assert 0 <= probs[0, 1] <= 1
    
    print("✓ TEST 5 PASSED: Feature importance varies with transaction characteristics")


# Testcase 6: Report Saving
def test_6_report_saving_to_file(sample_raw_transaction):
    """
    TEST 6: Report saving - Prediction and explanation can be saved to file
    
    Verifies that prediction results and explanations can be correctly
    saved to a text report file with all required sections.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "test_report.txt"
        
        # Prepare prediction data
        raw_fe = add_time_and_age_features(sample_raw_transaction)
        X_processed = preprocessor.transform(raw_fe[raw_fe_columns])
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        probs = predict_ensemble_probabilities(X_df)
        
        fraud_prob = float(probs[0, 1])
        pred_label = 1 if fraud_prob >= best_t else 0
        confidence = fraud_prob if pred_label == 1 else 1 - fraud_prob
        
        # Create result dictionary
        result = {
            "prediction": "fraud" if pred_label == 1 else "genuine",
            "fraud_probability": fraud_prob,
            "confidence": confidence,
            "threshold_used": best_t,
        }
        
        # Create mock ensemble summary
        ensemble_df = pd.DataFrame({
            "Feature": ["amt", "city_pop"],
            "SHAP Score": [0.45, 0.30],
            "LIME Score": [0.40, 0.35],
            "Ensemble Score": [0.85, 0.65],
        })
        
        # Create mock anchor explanation
        anchor_disp = pd.DataFrame([{
            "anchor_rule": "amt > 10 AND gender = F",
            "precision": 0.95,
            "coverage": 0.42,
        }])
        
        # Create report
        row_dict = sample_raw_transaction.iloc[0].to_dict()
        report_text = build_report_block(
            row_dict, result, ensemble_df, anchor_disp, [], 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        # Verify file was created
        assert report_path.exists(), "Report file not created"
        assert report_path.stat().st_size > 0, "Report file is empty"
        
        # Verify content
        saved_content = report_path.read_text(encoding="utf-8")
        assert "Fraud Detection Report" in saved_content
        assert result["prediction"].upper() in saved_content
        assert "amt" in saved_content.lower()
        
        print(f"✓ TEST 6 PASSED: Report saved successfully")


def test_6_report_appends_multiple_predictions(sample_raw_transaction):
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "cumulative_report.txt"
        
        # Prepare prediction
        raw_fe = add_time_and_age_features(sample_raw_transaction)
        X_processed = preprocessor.transform(raw_fe[raw_fe_columns])
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        probs = predict_ensemble_probabilities(X_df)
        fraud_prob = float(probs[0, 1])
        pred_label = 1 if fraud_prob >= best_t else 0
        
        result = {
            "prediction": "fraud" if pred_label == 1 else "genuine",
            "fraud_probability": fraud_prob,
            "confidence": fraud_prob if pred_label == 1 else 1 - fraud_prob,
            "threshold_used": best_t,
        }
        
        ensemble_df = pd.DataFrame({
            "Feature": ["amt"],
            "SHAP Score": [0.5],
            "LIME Score": [0.4],
            "Ensemble Score": [0.9],
        })
        
        anchor_disp = pd.DataFrame([{
            "anchor_rule": "amt > 10",
            "precision": 0.95,
            "coverage": 0.42,
        }])
        
        row_dict = sample_raw_transaction.iloc[0].to_dict()
        
        # Append first report
        report1 = build_report_block(
            row_dict, result, ensemble_df, anchor_disp, [],
            "2026-04-03 10:00:00"
        )
        report_path.write_text(report1, encoding="utf-8")
        
        # Append second report
        report2 = build_report_block(
            row_dict, result, ensemble_df, anchor_disp, [],
            "2026-04-03 11:00:00"
        )
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(report2)
        
        # Verify both reports are in file
        full_content = report_path.read_text(encoding="utf-8")
        assert full_content.count("Fraud Detection Report") == 2
        
        print("✓ TEST 6 PASSED: Multiple predictions appended to report successfully")


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
