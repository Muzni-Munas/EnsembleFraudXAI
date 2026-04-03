import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular
import dice_ml
from dice_ml import Dice

import streamlit as st
import json
import os
from datetime import datetime, date

# Loading the bundled model + assets
model_bundle_path = r".\Artifacts\model_bundle.pkl"
nn_model_path = r".\Artifacts\nn_model.pt"

with open(model_bundle_path, "rb") as f:
    bundle = pickle.load(f)

nn_payload = torch.load(nn_model_path, map_location="cpu")

print("Loaded bundle + nn payload")
print("Bundle keys:", list(bundle.keys()))
print("NN keys:", list(nn_payload.keys()))

# Unpacking bundle contents
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

dice_features_to_vary = bundle["dice_features_to_vary"]
permitted_range_base = bundle["permitted_range_base"]

xai_assets = bundle["xai_assets"]
shap_bg_df = xai_assets["shap_bg_df"]
lime_train_np = xai_assets["lime_train_np"]
anchor_bg_np = xai_assets["anchor_bg_np"]


# Rebuild MLP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

input_dim = int(nn_payload["input_dim"])
mlp_model = FraudMLP(input_dim).to(device)
mlp_model.load_state_dict(nn_payload["state_dict"])
mlp_model.eval()

print("MLP rebuilt ✅  | device =", device)


# Prediction helpers (for explainers + UI)
def predict_mlp_probabilities(X_df):
    mlp_model.eval()
    X_t = torch.tensor(X_df.to_numpy(), dtype=torch.float32).to(device)
    with torch.no_grad():
        p = torch.sigmoid(mlp_model(X_t)).cpu().numpy().ravel()
    return np.vstack([1 - p, p]).T

def predict_ensemble_probabilities(X_processed_df):
    rf_p   = rf_model.predict_proba(X_processed_df)[:, 1]
    xgb_p  = xgb_model.predict_proba(X_processed_df)[:, 1]
    lgbm_p = lgbm_model.predict_proba(X_processed_df)[:, 1]
    mlp_p  = predict_mlp_probabilities(X_processed_df)[:, 1]
    ens_p = (w_rf*rf_p + w_xgb*xgb_p + w_lgbm*lgbm_p + w_mlp*mlp_p)
    return np.vstack([1 - ens_p, ens_p]).T

def predict_ensemble_labels(X_np):
    X_df = pd.DataFrame(X_np, columns=feature_names)
    p = predict_ensemble_probabilities(X_df)[:, 1]
    return (p >= best_t).astype(int)


# Recreating SHAP (3 tree + MLP kernel)
shap_rf   = shap.TreeExplainer(rf_model, data=shap_bg_df)
shap_xgb  = shap.TreeExplainer(xgb_model, data=shap_bg_df)
shap_lgbm = shap.TreeExplainer(lgbm_model, data=shap_bg_df)

shap_mlp_bg_np = shap_bg_df.sample(200, random_state=42).to_numpy()

def mlp_predict_proba_from_numpy(X_np):
    X_df = pd.DataFrame(X_np, columns=feature_names)
    return predict_mlp_probabilities(X_df)

shap_mlp = shap.KernelExplainer(lambda x: mlp_predict_proba_from_numpy(x)[:, 1], shap_mlp_bg_np)

print("SHAP explainers recreated")


#  Recreating LIME
lime_explainer = LimeTabularExplainer(
    training_data=lime_train_np,
    feature_names=feature_names,
    class_names=["genuine", "fraud"],
    mode="classification",
    discretize_continuous=True
)
print("LIME explainer recreated")


# Recreating ANCHORS
anchor_explainer = AnchorTabular(
    predictor=predict_ensemble_labels,
    feature_names=feature_names,
    seed=42
)
anchor_explainer.fit(anchor_bg_np, disc_perc=(25, 50, 75))
print("Anchors explainer recreated")


# Recreating DiCE 
class EnsembleDiceWrapper:
    def __init__(self, preprocessor, processed_feature_names):
        self.preprocessor = preprocessor
        self.processed_feature_names = list(processed_feature_names)

    def predict_proba(self, X_raw_df):
        X_proc = self.preprocessor.transform(X_raw_df)
        X_proc_df = pd.DataFrame(X_proc, columns=self.processed_feature_names)
        return predict_ensemble_probabilities(X_proc_df)

dice_model = EnsembleDiceWrapper(preprocessor, feature_names)

dice_schema_df = bundle["dice_schema_df"].copy()

dice_data = dice_ml.Data(
    dataframe=dice_schema_df,
    continuous_features=[c for c in raw_fe_columns if c not in bundle["categorical_cols"]],
    categorical_features=bundle["categorical_cols"],
    outcome_name="is_fraud"
)

dice_model_obj = dice_ml.Model(model=dice_model, backend="sklearn", model_type="classifier")
dice_explainer = Dice(dice_data, dice_model_obj, method="random")

print("DiCE explainer recreated (schema based)")


# Helper for feature engineering 
def add_time_and_age_features(df):
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


# SHAP value extraction helper 
def get_absolute_shap_for_fraud_class(explainer, X_df, class_idx=1):
    sv = explainer.shap_values(X_df)
    if isinstance(sv, list):
        sv = sv[class_idx]
    sv = np.array(sv)

    if sv.ndim == 2 and sv.shape[0] == 1:
        vec = sv[0]
    elif sv.ndim == 2 and sv.shape[1] == 2:
        vec = sv[:, class_idx]
    elif sv.ndim == 3 and sv.shape[0] == 1 and sv.shape[-1] == 2:
        vec = sv[0, :, class_idx]
    elif sv.ndim == 1:
        vec = sv
    else:
        sv_s = np.squeeze(sv)
        if sv_s.ndim == 2 and sv_s.shape[1] == 2:
            vec = sv_s[:, class_idx]
        elif sv_s.ndim == 1:
            vec = sv_s
        else:
            raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

    return np.abs(vec).astype(float)

def get_per_model_shap_single_row(X_proc_df, nsamples_mlp=150):
    X_proc_df = X_proc_df[feature_names].copy()

    rf_abs   = get_absolute_shap_for_fraud_class(shap_rf,   X_proc_df, class_idx=1)
    xgb_abs  = get_absolute_shap_for_fraud_class(shap_xgb,  X_proc_df, class_idx=1)
    lgbm_abs = get_absolute_shap_for_fraud_class(shap_lgbm, X_proc_df, class_idx=1)

    mlp_sv = shap_mlp.shap_values(X_proc_df.to_numpy(), nsamples=nsamples_mlp)
    if isinstance(mlp_sv, list):
        mlp_sv = mlp_sv[0]
    mlp_sv = np.squeeze(np.array(mlp_sv))
    if mlp_sv.ndim == 2 and mlp_sv.shape[0] == 1:
        mlp_sv = mlp_sv[0]
    mlp_abs = np.abs(mlp_sv).astype(float)

    return {"rf": rf_abs, "xgb": xgb_abs, "lgbm": lgbm_abs, "mlp": mlp_abs}

def fuse_weighted_ensemble_shap(per_model_shap_abs):
    weights = {"rf": w_rf, "xgb": w_xgb, "lgbm": w_lgbm, "mlp": w_mlp}
    fused = np.zeros(len(feature_names), dtype=float)

    for k, arr in per_model_shap_abs.items():
        arr = np.array(arr, dtype=float)
        s = arr.sum()
        arr_norm = arr / s if s > 0 else arr
        fused += weights[k] * arr_norm

    return pd.DataFrame({"feature": feature_names, "shap_score": fused})


# Feature collapsing helpers 
def map_onehot_to_base_feature(f):
    f = str(f).replace("num__", "").replace("cat__", "")
    if f.startswith("category_"):
        return "category"
    if f.startswith("gender_"):
        return "gender"
    return f

def collapse_and_topk(df, feature_col, score_col, k=8):
    tmp = df[[feature_col, score_col]].copy()
    tmp["feature"] = tmp[feature_col].apply(map_onehot_to_base_feature)
    tmp = tmp.groupby("feature", as_index=False)[score_col].sum()
    tmp = tmp.sort_values(score_col, ascending=False).head(k).reset_index(drop=True)
    return tmp


# DiCE counterfactual change summary
def summarize_counterfactual_changes(original_raw_fe_df, cf_df, max_cfs=3):
    if cf_df is None or len(cf_df) == 0:
        return []
    orig = original_raw_fe_df.iloc[0]
    summaries = []
    for idx in range(min(max_cfs, len(cf_df))):
        cf = cf_df.iloc[idx]
        changed = []
        for c in original_raw_fe_df.columns:
            if c in cf.index and str(orig[c]) != str(cf[c]):
                changed.append({"feature": c, "from": orig[c], "to": cf[c]})
        summaries.append({"cf_id": idx + 1, "changes": changed})
    return summaries


# Predict + Explain 1 transaction
def predict_and_explain(raw_input_df, top_k=8, mlp_nsamples=150, anchor_threshold=0.95, dice_total_cfs=3):
    if not isinstance(raw_input_df, pd.DataFrame) or len(raw_input_df) != 1:
        raise ValueError("raw_input_df must be a single-row pandas DataFrame")

    # Applying FE
    raw_fe = add_time_and_age_features(raw_input_df)
    raw_fe = raw_fe[raw_fe_columns].copy()

    # Preprocessing - processed space for models
    X_proc = preprocessor.transform(raw_fe)
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

    # Ensemble prediction
    ens_proba = float(predict_ensemble_probabilities(X_proc_df)[0, 1])
    pred_label = int(ens_proba >= float(best_t))
    confidence = float(ens_proba if pred_label == 1 else 1 - ens_proba)

    # SHAP (fusing all models)
    per_model = get_per_model_shap_single_row(X_proc_df, nsamples_mlp=mlp_nsamples)
    shap_df = fuse_weighted_ensemble_shap(per_model)
    shap_top = collapse_and_topk(shap_df, "feature", "shap_score", k=top_k)

    # LIME
    lime_exp = lime_explainer.explain_instance(
        X_proc_df.to_numpy().ravel(),
        predict_fn=lambda x: predict_ensemble_probabilities(pd.DataFrame(x, columns=feature_names)),
        num_features=top_k
    )
    lime_rows = [{"feature": map_onehot_to_base_feature(rule), "lime_score": abs(w)} for rule, w in lime_exp.as_list()]
    lime_df = pd.DataFrame(lime_rows)
    lime_top = lime_df.groupby("feature", as_index=False)["lime_score"].sum().sort_values("lime_score", ascending=False).head(top_k)

    # Anchors
    anchor_exp = anchor_explainer.explain(X_proc_df.to_numpy()[0], threshold=anchor_threshold, seed=42)
    anchor_rule = " AND ".join(anchor_exp.data.get("anchor", [])).strip()
    anchor_out = {
        "anchor_rule": anchor_rule if anchor_rule else None,
        "precision": float(anchor_exp.data["precision"]),
        "coverage": float(anchor_exp.data["coverage"]),
        "threshold": float(anchor_threshold)
    }

    # DiCE
    exp = dice_explainer.generate_counterfactuals(
        query_instances=raw_fe,
        total_CFs=dice_total_cfs,
        desired_class="opposite",
        features_to_vary=bundle["dice_features_to_vary"],
        permitted_range=bundle["permitted_range_base"]
    )
    try:
        cf_df = exp.cf_examples_list[0].final_cfs_df.copy()
        cf_df = cf_df[raw_fe_columns]
    except Exception:
        cf_df = pd.DataFrame(columns=raw_fe_columns)

    dice_summary = summarize_counterfactual_changes(raw_fe, cf_df, max_cfs=dice_total_cfs)

    # Final explanation fusion (SHAP + LIME)
    shap_norm = shap_top.copy()
    lime_norm = lime_top.copy()

    shap_norm["shap_score"] = shap_norm["shap_score"] / (shap_norm["shap_score"].sum() + 1e-9)
    lime_norm["lime_score"] = lime_norm["lime_score"] / (lime_norm["lime_score"].sum() + 1e-9)

    merged = shap_norm.merge(lime_norm, on="feature", how="outer").fillna(0.0)
    merged["ensemble_expl_score"] = 0.6 * merged["shap_score"] + 0.4 * merged["lime_score"]
    merged = merged.sort_values("ensemble_expl_score", ascending=False).head(top_k).reset_index(drop=True)

    return {
        "prediction": "fraud" if pred_label == 1 else "genuine",
        "fraud_probability": ens_proba,
        "confidence": confidence,
        "threshold_used": float(best_t),
        "explanations": {
            "final_ensemble_summary": merged.to_dict(orient="records"),
            "shap_top": shap_top.to_dict(orient="records"),
            "lime_top": lime_top.to_dict(orient="records"),
            "anchor": anchor_out,
            "dice": {
                "total_found": int(len(cf_df)),
                "changes_summary": dice_summary
            }
        }
    }



# UI implementation using Streamlit
 
st.set_page_config(page_title="Ensemble XAI Fraud Prototype", layout="wide")
 
st.title("Ensemble Explainable Fraud Detection")
st.write("Fill in the transaction details below and click **Predict & Explain**.")
 
# Report file path (to save the details))
REPORT_PATH = "./Reports/Report.txt"

 
# Applying custom css to ensure text input labels
st.markdown(
    """
    <style>
    div[data-testid="stTextInput"] label{
        opacity: 1 !important;
        color: inherit !important;
        -webkit-text-fill-color: inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# Automatically fill the transaction date & time
if "trans_dt" not in st.session_state:
    st.session_state["trans_dt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
# Input form to get transaction details from the user
with st.form("transaction_form"):
    st.subheader("Transaction Details")
 
    col1, col2 = st.columns(2)
 
    with col1:
        trans_date_trans_time = st.text_input(
            "Transaction Date & Time",
            value=st.session_state["trans_dt"],
            help="Auto-filled with current time. Format: YYYY-MM-DD HH:MM:SS",
            disabled=True,
        )
 
        category = st.text_input(
            "Category",
            value="personal_care",
            help="e.g. personal_care, grocery_pos, entertainment, shopping_net",
        )
 
        amt = st.number_input(
            "Amount (USD)",
            min_value=0.0,
            value=20.86,
            step=0.01,
            format="%.2f",
            help="Transaction amount in USD",
        )
 
        gender = st.selectbox(
            "Gender",
            options=["Female", "Male"],
            index=0,
            help="Cardholder gender",
        )
        gender_value = "F" if gender == "Female" else "M"
 
        dob = st.date_input(
            "Date of Birth",
            value=date(1968, 3, 19),
            min_value=date(1926, 1, 1),
            max_value=date.today(),
            help="Cardholder date of birth",
        )
 
    with col2:
        lat = st.number_input(
            "Cardholder Latitude",
            value=33.9659,
            format="%.6f",
            help="Latitude of cardholder's location",
        )
 
        long_ = st.number_input(
            "Cardholder Longitude",
            value=-80.9355,
            format="%.6f",
            help="Longitude of cardholder's location",
        )
 
        city_pop = st.number_input(
            "City Population",
            min_value=0,
            value=333497,
            step=1,
            help="Population of the cardholder's city",
        )
 
        merch_lat = st.number_input(
            "Merchant Latitude",
            value=33.986391,
            format="%.6f",
            help="Latitude of merchant location (auto-filled from device if available)",
        )
 
        merch_long = st.number_input(
            "Merchant Longitude",
            value=-81.200714,
            format="%.6f",
            help="Longitude of merchant location (auto-filled from device if available)",
        )
 
    st.divider()
    run_btn = st.form_submit_button(
        "Predict & Explain", type="primary", width="stretch"
    )
 
 
# Table for ensemble summary (top-3 by combined SHAP+LIME score)
def build_ensemble_summary(shap_top_df, lime_top_df, top_n_each=5, final_top=3):
    shap_top = shap_top_df.head(top_n_each).copy()
    lime_top = lime_top_df.head(top_n_each).copy()
 
    shap_top = shap_top.rename(
        columns={shap_top.columns[0]: "feature", shap_top.columns[1]: "shap_score"}
    )
    lime_top = lime_top.rename(
        columns={lime_top.columns[0]: "feature", lime_top.columns[1]: "lime_score"}
    )
 
    merged = pd.merge(shap_top, lime_top, on="feature", how="outer").fillna(0.0)
    merged["ensemble_score"] = merged["shap_score"] + merged["lime_score"]
    merged = (
        merged.sort_values("ensemble_score", ascending=False)
        .head(final_top)
        .reset_index(drop=True)
    )
    merged.columns = ["Feature", "SHAP Score", "LIME Score", "Ensemble Score"]
    merged["SHAP Score"]     = merged["SHAP Score"].round(6)
    merged["LIME Score"]     = merged["LIME Score"].round(6)
    merged["Ensemble Score"] = merged["Ensemble Score"].round(6)
    return merged
 
 
# Table for DiCE counterfactual changes (filtered to only show changes of top features)
def filter_dice_to_top_features(dice_changes, top_features):
    seen = set()
    rows = []
    for cf in dice_changes:
        cf_id = cf.get("cf_id")
        for ch in cf.get("changes", []):
            feat = ch.get("feature")
            if feat not in top_features:
                continue
            from_val = str(ch.get("from", ""))
            to_val   = str(ch.get("to",   ""))
            key = (feat, from_val, to_val)   # deduplicate on content, not CF id
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "CF #":    cf_id,
                "Feature": feat,
                "From":    from_val,
                "To":      to_val,
            })
    return rows
 
 
# Formatting as text block for the report file
def build_report_block(row_dict, result, ensemble_df, anchor_disp, dice_rows, timestamp):
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
        json.dumps(row_dict, indent=2),
        "",
        "--- Final Ensemble Summary of Influential Attributes (Top 3) ---",
        f"{'Feature':<20} {'SHAP Score':>12} {'LIME Score':>12} {'Ensemble Score':>15}",
        "-" * 63,
    ]
    for _, r in ensemble_df.iterrows():
        lines.append(
            f"{r['Feature']:<20} {r['SHAP Score']:>12.6f} "
            f"{r['LIME Score']:>12.6f} {r['Ensemble Score']:>15.6f}"
        )
    lines += [
        "",
        "--- Anchor Explanation ---",
        f"Rule      : {anchor_disp['anchor_rule'].iloc[0]}",
        f"Precision : {anchor_disp['precision'].iloc[0]:.4f}",
        f"Coverage  : {anchor_disp['coverage'].iloc[0]:.4f}",
        "",
        "--- DiCE Counterfactual Changes (Top Features Only) ---",
    ]
    if dice_rows:
        lines.append(f"{'CF #':<6} {'Feature':<20} {'From':<20} {'To':<20}")
        lines.append("-" * 68)
        for r in dice_rows:
            lines.append(
                f"{r['CF #']:<6} {r['Feature']:<20} {r['From']:<20} {r['To']:<20}"
            )
    else:
        lines.append("No counterfactual changes found for top features.")
    lines.append("\n")
    return "\n".join(lines)
 
 
# Generating prediction & display results
if run_btn:
    row_dict = {
        "trans_date_trans_time": trans_date_trans_time,
        "category": category.strip(),
        "amt": float(amt),
        "gender": gender_value,
        "lat": float(lat),
        "long": float(long_),
        "city_pop": int(city_pop),
        "dob": dob.strftime("%Y-%m-%d"),
        "merch_lat": float(merch_lat),
        "merch_long": float(merch_long),
    }

    try:
        raw_input_df = pd.DataFrame([row_dict])
        result = predict_and_explain(raw_input_df)

        st.success("Prediction completed ✅")

        # Prediction metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Label", result["prediction"].upper())
        c2.metric("Fraud Probability", f'{result["fraud_probability"]:.6f}')
        c3.metric("Confidence", f'{result["confidence"]:.6f}')

        st.divider()

        # Final ensemble summary of influential attributes (Top 3)
        shap_top_raw = pd.DataFrame(result["explanations"]["shap_top"])
        lime_top_raw = pd.DataFrame(result["explanations"]["lime_top"])

        for df_ in [shap_top_raw, lime_top_raw]:
            df_.iloc[:, 0] = df_.iloc[:, 0].apply(map_onehot_to_base_feature)

        ensemble_df = build_ensemble_summary(
            shap_top_raw, lime_top_raw, top_n_each=5, final_top=3
        )
        top_features = set(ensemble_df["Feature"].tolist())

        st.subheader("Final Ensemble Summary of Influential Attributes")
        st.dataframe(ensemble_df, width="stretch", hide_index=True)

        st.divider()

        # Anchor explanation
        st.subheader("Anchor Explanation")
        anchor_disp = pd.DataFrame([result["explanations"]["anchor"]])
        if "anchor_rule" in anchor_disp.columns:
            anchor_disp["anchor_rule"] = (
                anchor_disp["anchor_rule"]
                .astype(str)
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )
        st.dataframe(anchor_disp, width="stretch", hide_index=True)

        st.divider()

        # DiCE counterfactual changes
        st.subheader("DiCE Counterfactual Changes")
        dice_changes = result["explanations"]["dice"]["changes_summary"]
        dice_rows = filter_dice_to_top_features(dice_changes, top_features)

        if not dice_rows:
            st.info("No counterfactual changes found for the top influential features.")
        else:
            dice_display_df = pd.DataFrame(dice_rows)
            dice_display_df["From"] = dice_display_df["From"].astype(str)
            dice_display_df["To"] = dice_display_df["To"].astype(str)
            st.dataframe(dice_display_df, width="stretch", hide_index=True)

        st.divider()

        # Storing report text in session
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_block = build_report_block(
            row_dict, result, ensemble_df, anchor_disp, dice_rows, timestamp_str
        )
        st.session_state["last_report_block"] = report_block

    except Exception as e:
        st.error(f"Error: {e}")


if "last_report_block" in st.session_state:
    st.subheader("Report")

    report_block = st.session_state["last_report_block"]

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("Save to Report", width="stretch", key="save_report_btn"):
            try:
                report_dir = os.path.dirname(REPORT_PATH)
                if report_dir:
                    os.makedirs(report_dir, exist_ok=True)

                with open(REPORT_PATH, "a", encoding="utf-8") as f:
                    f.write(report_block)

                st.success(f"Successfully saved to report")

            except Exception as e:
                st.error(f"Error saving report: {e}")

    with btn_col2:
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                download_data = f.read()
        else:
            download_data = report_block

        st.download_button(
            label="Download Full Report (.txt)",
            data=download_data,
            file_name="Report.txt",
            mime="text/plain",
            width="stretch",
            key="download_report_btn",
        )

#py -m streamlit run .\UI\app.py