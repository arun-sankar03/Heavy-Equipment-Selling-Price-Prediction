# Heavy-Equipment-Selling-Price-Prediction
Kaggle Tabular Regression pipeline featuring ordinal feature engineering, and an ensemble blend of XGBoost, LightGBM, and CatBoost to minimize RMSLE.
An end-to-end Machine Learning pipeline built for a Kaggle tabular regression competition. This solution leverages advanced feature engineering, target log-transformation, ensemble blending **XGBoost**, **LightGBM**, and **CatBoost** to optimize root mean squared logarithmic error (RMSLE).

---

## 📌 Project Overview & Highlights

* **Objective:** Predict numerical target values while minimizing RMSLE loss.
* **Feature Engineering:** Extracted domain interactions (e.g., total expected asset usage hours) and length anomalies from descriptors.
* **Categorical Encoding:** Optimized for tree-based split histograms using `OrdinalEncoder` over high-sparsity One-Hot techniques.
* **Target Transformation:** Applied `np.log1p` during training and `np.expm1` for inference to align optimization directly with RMSLE.
* **Model Ensembling:** 5-Fold Cross-Validation blending predictions across three top-tier gradient boosting frameworks:
  * **XGBoost Regressor** (Weight: 0.30)
  * **LightGBM Regressor** (Weight: 0.30)
  * **CatBoost Regressor** (Weight: 0.40)

---

## 📊 Pipeline Architecture

```text
Raw Data ➔ Ordinal Encoding ➔ Log1p Target ➔ 5-Fold Split
                                                   │
     ┌─────────────────────────────────────────────┼─────────────────────────────────────────────┐
     ▼                                             ▼                                             ▼
XGBoost Regressor                             LightGBM Regressor                            CatBoost Regressor
     │                                             │                                             │
     └─────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                                   ▼
                                     Weighted Ensemble Prediction
                                                   ▼
                                         np.expm1 Inverse Log
                                                   ▼
                                            Final Submission
```

---
