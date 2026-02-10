# train_models.py - training all classifiers on online shoppers dataset
# dataset: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)


# ===================== DATA LOADING & PREPROCESSING =====================

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    
    try:
        data = pd.read_csv(url)
        print(f"Downloaded dataset. Shape: {data.shape}")
    except:
        # if download fails try local copy
        data = pd.read_csv("online_shoppers_intention.csv")
        print(f"Loaded local dataset. Shape: {data.shape}")
    
    data.to_csv("online_shoppers_intention.csv", index=False)
    return data


def preprocess(data):
    print(f"\nFeatures: {data.shape[1] - 1}, Instances: {data.shape[0]}")
    print(f"Target distribution:\n{data['Revenue'].value_counts()}")
    
    # label encode the categorical features
    encoders = {}
    cat_cols = ['Month', 'VisitorType']
    for c in cat_cols:
        enc = LabelEncoder()
        data[c] = enc.fit_transform(data[c].astype(str))
        encoders[c] = enc
    
    # boolean to int
    data['Weekend'] = data['Weekend'].astype(int)
    data['Revenue'] = data['Revenue'].astype(int)
    
    X = data.drop('Revenue', axis=1)
    y = data['Revenue']
    
    # 80-20 split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # standardize features
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    
    # save preprocessing objects
    pickle.dump(sc, open('model/scaler.pkl', 'wb'))
    pickle.dump(encoders, open('model/label_encoders.pkl', 'wb'))
    pickle.dump({'feature_names': list(X.columns), 'categorical_cols': cat_cols},
                open('model/feature_info.pkl', 'wb'))
    
    # save test set separately for streamlit upload
    test_set = X_test.copy()
    test_set['Revenue'] = y_test.values
    test_set.to_csv("test_data.csv", index=False)
    print("Test data saved to test_data.csv\n")
    
    return X_train_sc, X_test_sc, y_train, y_test


# ===================== METRICS CALCULATION =====================

def get_metrics(y_actual, y_predicted, y_proba):
    return {
        'Accuracy': round(accuracy_score(y_actual, y_predicted), 4),
        'AUC': round(roc_auc_score(y_actual, y_proba), 4),
        'Precision': round(precision_score(y_actual, y_predicted, zero_division=0), 4),
        'Recall': round(recall_score(y_actual, y_predicted, zero_division=0), 4),
        'F1': round(f1_score(y_actual, y_predicted, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_actual, y_predicted), 4)
    }


# ===================== MAIN TRAINING LOOP =====================

def run_training():
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess(data)
    
    # all 6 classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'kNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(
            n_estimators=150, random_state=42, max_depth=15
        ),
        'XGBoost (Ensemble)': XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss', use_label_encoder=False
        )
    }
    
    all_results = {}
    
    print("=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)
    
    for model_name, clf in classifiers.items():
        print(f"\n>> {model_name}")
        
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        scores = get_metrics(y_test, preds, probs)
        all_results[model_name] = scores
        
        for k, v in scores.items():
            print(f"   {k}: {v}")
        
        cm = confusion_matrix(y_test, preds)
        print(f"   Confusion Matrix: {cm[0]} / {cm[1]}")
        
        # save model
        fname = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        pickle.dump(clf, open(f'model/{fname}.pkl', 'wb'))
        print(f"   Saved -> model/{fname}.pkl")
    
    # comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    
    df_results = pd.DataFrame(all_results).T
    df_results.index.name = 'Model'
    print(df_results.to_string())
    
    # save for streamlit app
    pickle.dump(all_results, open('model/results.pkl', 'wb'))
    df_results.to_csv('model/comparison_table.csv')
    print("\nDone! All models saved.")


if __name__ == "__main__":
    run_training()
