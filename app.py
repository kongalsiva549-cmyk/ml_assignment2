import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, pathlib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# set working dir to where app.py lives (needed for streamlit cloud)
os.chdir(pathlib.Path(__file__).parent)

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛍️", layout="wide")

# --- load saved stuff ---
@st.cache_resource
def get_model(name):
    fname = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    return pickle.load(open(f'model/{fname}.pkl', 'rb'))

@st.cache_resource
def get_artifacts():
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    encoders = pickle.load(open('model/label_encoders.pkl', 'rb'))
    feat_info = pickle.load(open('model/feature_info.pkl', 'rb'))
    results = pickle.load(open('model/results.pkl', 'rb'))
    return scaler, encoders, feat_info, results


def process_input(df, scaler, encoders, feat_info):
    """apply same preprocessing as training"""
    temp = df.copy()
    
    for col in feat_info['categorical_cols']:
        if col in temp.columns:
            le = encoders[col]
            temp[col] = temp[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
            )
    
    if 'Weekend' in temp.columns:
        temp['Weekend'] = temp['Weekend'].astype(int)
    
    y = None
    if 'Revenue' in temp.columns:
        temp['Revenue'] = temp['Revenue'].astype(int)
        y = temp['Revenue']
        temp = temp.drop('Revenue', axis=1)
    
    # reorder columns to match training
    for f in feat_info['feature_names']:
        if f not in temp.columns:
            temp[f] = 0
    temp = temp[feat_info['feature_names']]
    
    X = scaler.transform(temp)
    return X, y


def show_cm(y_true, y_pred, name):
    """confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['No Purchase', 'Purchase'],
                yticklabels=['No Purchase', 'Purchase'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'{name}')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN UI
# ============================================================

def main():
    st.title("🛍️ Online Shoppers Purchase Prediction")
    st.write("Predict whether a user session will result in a purchase using 6 different ML models. "
             "Dataset from UCI ML Repository (12,330 sessions, 17 features).")
    
    try:
        scaler, encoders, feat_info, saved_results = get_artifacts()
    except FileNotFoundError:
        st.error("Model files missing. Run model/train_models.py first!")
        return
    
    # --- sidebar ---
    st.sidebar.title("Settings")
    
    models_list = ['Logistic Regression', 'Decision Tree', 'kNN',
                   'Naive Bayes', 'Random Forest (Ensemble)', 'XGBoost (Ensemble)']
    
    chosen_model = st.sidebar.selectbox("Pick a model:", models_list, index=5)
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Dataset Info**\n\n"
                    "- Source: UCI Repository\n"
                    "- 12,330 sessions\n"
                    "- 17 features\n"
                    "- Binary target (Revenue)")
    
    # --- file upload ---
    st.subheader("📂 Upload Test Data (CSV)")
    uploaded = st.file_uploader("Upload CSV with same features as training data", type=['csv'])
    
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        
        st.write(f"**Loaded:** {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(8))
        
        has_label = 'Revenue' in df.columns
        if not has_label:
            st.warning("No 'Revenue' column found - will show predictions only, no evaluation metrics.")
        
        try:
            X_input, y_true = process_input(df, scaler, encoders, feat_info)
        except Exception as ex:
            st.error(f"Preprocessing error: {ex}")
            return
        
        model = get_model(chosen_model)
        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:, 1]
        
        st.markdown("---")
        st.subheader(f"Results — {chosen_model}")
        
        if has_label:
            # compute all 6 metrics
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc_val = matthews_corrcoef(y_true, y_pred)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.4f}")
            c1.metric("AUC", f"{auc:.4f}")
            c2.metric("Precision", f"{prec:.4f}")
            c2.metric("Recall", f"{rec:.4f}")
            c3.metric("F1 Score", f"{f1:.4f}")
            c3.metric("MCC", f"{mcc_val:.4f}")
            
            st.markdown("---")
            
            left, right = st.columns(2)
            with left:
                st.write("**Confusion Matrix**")
                fig = show_cm(y_true, y_pred, chosen_model)
                st.pyplot(fig)
            
            with right:
                st.write("**Classification Report**")
                rpt = classification_report(y_true, y_pred,
                                           target_names=['No Purchase', 'Purchase'],
                                           output_dict=True)
                st.dataframe(pd.DataFrame(rpt).T.style.format("{:.4f}"))
        else:
            # no labels - just show predictions
            out = df.copy()
            out['Prediction'] = y_pred
            out['Purchase_Prob'] = np.round(y_prob, 4)
            st.dataframe(out)
        
        # download button
        st.markdown("---")
        dl = df.copy()
        dl['Prediction'] = y_pred
        dl['Purchase_Prob'] = np.round(y_prob, 4)
        st.download_button("⬇️ Download Results CSV",
                          dl.to_csv(index=False).encode('utf-8'),
                          "predictions.csv", "text/csv")
    
    else:
        st.info("Upload a CSV above to see predictions. You can use the test_data.csv file from the repo.")
    
    # --- always show comparison table ---
    st.markdown("---")
    st.subheader("📊 Model Comparison (All 6 Classifiers)")
    
    comp = pd.DataFrame(saved_results).T
    comp.index.name = 'Model'
    st.dataframe(comp.style.highlight_max(axis=0, color='#90EE90')
                           .highlight_min(axis=0, color='#FFCCCB')
                           .format("{:.4f}"))
    
    # comparison bar chart
    st.write("**Visual Comparison**")
    metric_choice = st.selectbox("Choose metric:", comp.columns.tolist())
    
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    clrs = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948']
    bars = ax2.bar(comp.index, comp[metric_choice], color=clrs, edgecolor='gray', linewidth=0.5)
    ax2.set_ylabel(metric_choice)
    ax2.set_title(f'{metric_choice} - All Models')
    ax2.set_ylim(0, 1.08)
    plt.xticks(rotation=25, ha='right', fontsize=9)
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2., b.get_height() + 0.015,
                f'{b.get_height():.3f}', ha='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)


if __name__ == "__main__":
    main()
