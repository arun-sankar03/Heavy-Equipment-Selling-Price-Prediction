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

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
   cd YOUR-REPO-NAME
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Data:**
   Place `train.csv`, `test.csv`, and `sample_submission.csv` inside a `data/` directory.

4. **Execute Notebook:**
   Launch Jupyter Notebook or Jupyter Lab to run the end-to-end training and inference script:
   ```bash
   jupyter notebook notebooks/solution_pipeline.ipynb
   ```

---

## 📈 Model Performance & Results

* **Baseline Single Model (Random Forest / Basic Split):** ~0.211 RMSLE
* **5-Fold CV Blended Ensemble (XGB + LGB + CatBoost):** **~0.190 RMSLE**
