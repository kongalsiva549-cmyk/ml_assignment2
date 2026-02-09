# Online Shoppers Purchasing Intention Prediction

## Problem Statement

Online shopping platforms generate a huge amount of user session data. Understanding whether a visitor will actually complete a purchase is crucial for businesses to optimize their marketing strategies, improve user experience, and increase revenue. This project builds and compares multiple Machine Learning classification models to predict whether an online shopping session will end in a purchase (revenue generation) or not, based on various session attributes like page views, duration, bounce rates, and user demographics.

## Dataset Description

- **Dataset Name:** Online Shoppers Purchasing Intention Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
- **Number of Instances:** 12,330
- **Number of Features:** 17
- **Target Variable:** `Revenue` (Binary: True/False)
- **Class Distribution:** ~84.5% No Purchase, ~15.5% Purchase (imbalanced)

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| Administrative | Number of administrative pages visited |
| Administrative_Duration | Total time spent on administrative pages |
| Informational | Number of informational pages visited |
| Informational_Duration | Total time spent on informational pages |
| ProductRelated | Number of product-related pages visited |
| ProductRelated_Duration | Total time spent on product-related pages |
| BounceRates | Average bounce rate of pages visited |
| ExitRates | Average exit rate of pages visited |
| PageValues | Average page value of pages visited |
| SpecialDay | Closeness of visit to a special day |
| Month | Month of the visit |
| OperatingSystems | Operating system of the visitor |
| Browser | Browser used by the visitor |
| Region | Geographic region of the visitor |
| TrafficType | Traffic source type |
| VisitorType | Whether returning, new, or other visitor |
| Weekend | Whether the visit was on a weekend |

## Models Used

Six classification models were implemented and evaluated on this dataset:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbor (kNN) Classifier**
4. **Naive Bayes (Gaussian)**
5. **Random Forest (Ensemble)**
6. **XGBoost (Ensemble)**

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8832 | 0.8653 | 0.7640 | 0.3560 | 0.4857 | 0.4696 |
| Decision Tree | 0.8804 | 0.8315 | 0.6276 | 0.5602 | 0.5920 | 0.5233 |
| kNN | 0.8783 | 0.7990 | 0.6990 | 0.3770 | 0.4898 | 0.4540 |
| Naive Bayes | 0.7794 | 0.8020 | 0.3802 | 0.6728 | 0.4858 | 0.3826 |
| Random Forest (Ensemble) | 0.8990 | 0.9168 | 0.7301 | 0.5524 | 0.6289 | 0.5792 |
| XGBoost (Ensemble) | 0.9011 | 0.9242 | 0.7226 | 0.5864 | 0.6474 | 0.5949 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy is 88.32% which seems good, but recall is only 0.356 — it misses nearly 64% of actual purchasers. This is because the model is linear and the dataset is heavily imbalanced (84.5% non-purchasers). The high precision (0.764) means when it predicts a purchase, it is usually correct, but it is too conservative overall. Suitable as a simple baseline. |
| Decision Tree | Recall jumps to 0.5602 which is the best among individual (non-ensemble) models — it catches more true buyers by learning non-linear splits. However, AUC is the lowest across all models at 0.8315, and precision drops to 0.6276 indicating more false positives. Even with max_depth=10, the single tree overfits compared to ensembles. |
| kNN | Accuracy (87.83%) and recall (0.377) are similar to Logistic Regression. With 17 features the distance-based approach suffers from the curse of dimensionality. AUC is the second lowest at 0.799. The model is also sensitive to the value of k and does not generalize well on this dataset. |
| Naive Bayes | Lowest accuracy (77.94%) but highest recall (0.6728) among all individual models — it flags most actual purchasers. The trade-off is poor precision (0.3802), generating 419 false positives out of 2084 non-purchase sessions. The independence assumption between correlated features like BounceRates and ExitRates hurts overall performance significantly. |
| Random Forest (Ensemble) | Clear improvement over single models — 89.90% accuracy, 0.9168 AUC, and balanced precision-recall trade-off. The ensemble of 150 bagged trees reduces the overfitting seen in the single Decision Tree while maintaining decent recall (0.5524). MCC of 0.5792 confirms good overall classification quality on this imbalanced dataset. |
| XGBoost (Ensemble) | Best model across the board — highest accuracy (90.11%), AUC (0.9242), F1 (0.6474), and MCC (0.5949). Boosting sequentially corrects misclassified samples, which helps learn the minority purchase class. It achieves the best precision-recall balance (0.7226 / 0.5864). Built-in L1/L2 regularization prevents overfitting even with 150 estimators. |

## Project Structure

```
ml_assignment2/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── test_data.csv             # Sample test data for the app
├── online_shoppers_intention.csv  # Full dataset
└── model/
    ├── train_models.py       # Training script for all 6 models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest_ensemble.pkl
    ├── xgboost_ensemble.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    ├── feature_info.pkl
    ├── results.pkl
    └── comparison_table.csv
```

## How to Run Locally

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ml_assignment2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models (first time only):
```bash
cd model
python train_models.py
cd ..
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Streamlit App Features

- **Dataset Upload:** Upload test data in CSV format for evaluation
- **Model Selection:** Choose from 6 different ML models via dropdown
- **Evaluation Metrics:** View Accuracy, AUC, Precision, Recall, F1, and MCC scores
- **Confusion Matrix:** Visual heatmap of the confusion matrix
- **Classification Report:** Detailed per-class precision, recall, and F1 scores
- **Model Comparison:** Side-by-side comparison of all models with interactive bar charts
- **Download Predictions:** Export predictions as CSV file

## Deployment

The app is deployed on Streamlit Community Cloud.

**Live App Link:** [Click here to access the app](https://mlassignment2-ecf32r6k96hvglm7vj8lyy.streamlit.app/)

## Technologies Used

- Python 3.9+
- Scikit-learn
- XGBoost
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
