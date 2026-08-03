# Pakistan-E-commerce-Order-Status-Prediction---Model
Pakistan E-commerce Industry is  evolving through different phases. And till people trust  is fluctuating due to many circumstances that might effect industry credibility . Due to this order (Complete cancel) ratio is still not stable and retailers have to face this issue . This model is to predict the order is more expected to cancel or to complete . 


# 📦 Pakistan E-commerce Order Cancellation Prediction

Predicting whether an order will be **Completed** or **Cancelled** using order-level features from a large-scale Pakistani e-commerce transactions dataset (500K+ orders).

---

## 🎯 Project Overview

E-commerce platforms lose significant revenue and operational efficiency to cancelled orders. This project builds a machine learning classifier that predicts order outcome (Completed vs Cancelled) at the time of order placement, using features available before the order is fulfilled — enabling proactive intervention on high-risk orders.

**Key result:** XGBoost classifier achieving **75% accuracy** and **0.75 F1-score**, with balanced performance across both classes.

---

## 📁 Repository Structure

| File | Purpose |
|---|---|
| `data_cleaning.ipynb` | Data cleaning, outlier detection, and feature engineering |
| `model.ipynb` | Model selection, training, tuning, and evaluation |
| `cancellation_predictor_app.py` | Interactive Streamlit app for live predictions |
| `README.md` | This file |

---

## 🧹 Part 1: Data Cleaning (`data_cleaning.ipynb`)

This notebook takes the raw transaction data and prepares it for modeling.

**What it covers:**
- **Handling NaNs / missing values** — identifying and treating missing entries across key columns
- **Outlier detection** — using IQR/statistical methods, applied *within category groups* (since price ranges vary drastically between e.g. Electronics vs Fashion)
- **Feature engineering** — deriving new, more useful columns from raw data:
  - `Shopping_Season` — grouping order months into Summer / Spring / Winter to capture seasonal cancellation patterns
  - Date decomposition — splitting order timestamps into `Order_year`, `Order_month`, `Order_day`
  - Consistency checks between `price`, `qty_ordered`, `grand_total`, `Total_before_disc`, and `discount_amount`
- **Redundancy handling** — since `grand_total = Total_before_disc − discount_amount`, only two of these three columns are kept (same logic as the dummy variable trap: k related columns → keep k−1 independent ones)

**Output:** a cleaned dataset ready for modeling, with all engineered features included.

---

## 🤖 Part 2: Modeling (`model.ipynb`)

This notebook builds, tunes, and evaluates the classification model.

### Target Variable
- Original `status` column had 5 classes: `Cancelled`, `Completed`, `Inprocess`, `Fraud`, `Unknown`
- `Inprocess` (not yet a final outcome), `Fraud`, and `Unknown` (statistically negligible) were dropped
- Final binary target: `is_completed` → **1 = Completed, 0 = Cancelled**, encoded via `LabelEncoder`

### Feature Set

| Column | Type | Notes |
|---|---|---|
| `price` | Numeric | Item price |
| `qty_ordered` | Numeric | Quantity in order |
| `grand_total` | Numeric | Final amount charged (after discount) |
| `discount_amount` | Numeric | Discount applied |
| `category_name_1` | Categorical → One-Hot Encoded | Product category (5 categories) |
| `payment_method` | Categorical → One-Hot Encoded | COD / Online Transfer / Pay later / Other |
| `Order_year` | Numeric | Year order was placed |
| `Order_month` | Numeric | Month order was placed |
| `Order_day` | Numeric | Day of month order was placed |

**Categorical encoding:** `pd.get_dummies()` with `drop_first=True` to avoid the dummy variable trap (multicollinearity from perfectly correlated dummy columns).

**Scaling:** Not applied — tree-based models (XGBoost) split on thresholds rather than distances, so feature scaling doesn't affect performance.

### Model Selection & Tuning
- **Algorithm:** XGBoost Classifier (`tree_method='hist'` for speed on large data)
- **Tuning method:** `RandomizedSearchCV` (faster than exhaustive `GridSearchCV` on 500K+ rows), with `cv=3` and `n_jobs=-1`
- Initial tuning run on a stratified sample for speed, then the best hyperparameters were used to refit on the **full training set**

### Train/Test Split
- 80/20 split, **stratified** on the target to preserve class balance (~53% Cancelled / 47% Completed) in both sets

---

## 📊 Results

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (Cancelled) | 0.78 | 0.74 | 0.76 | 53,756 |
| 1 (Completed) | 0.72 | 0.77 | 0.74 | 47,060 |
| **Accuracy** | | | **0.75** | 100,816 |
| **Macro avg** | 0.75 | 0.75 | 0.75 | 100,816 |
| **Weighted avg** | 0.75 | 0.75 | 0.75 | 100,816 |

### Key Insight: Feature Importance

| Feature | Importance |
|---|---|
| `payment_method_Online Transfer` | **0.599** |
| `Order_year` | 0.054 |
| `Order_month` | 0.047 |
| `discount_amount` | 0.045 |
| `category_name_1_Other` | 0.042 |
| `price` | 0.023 |
| `grand_total` | 0.022 |

**Payment method dominates the model.** Orders paid via **Online Transfer cancel far more often than COD orders**:
- Cancelled orders: 71.1% Online Transfer vs 28.3% COD
- Completed orders: 63.3% COD vs 34.8% Online Transfer

This is a strong, actionable business insight — likely tied to payment verification friction on Online Transfer orders (failed/delayed transfers leading to automatic cancellation), rather than pure customer behavior.

**Other confirmed patterns from EDA:**
- **Category:** Electronics & Tech is over-represented in cancellations (37.6% of cancelled orders vs 29.1% of completed orders)
- **Price:** Cancelled orders have a higher median price (Rs. 1,299) than Completed orders (Rs. 749)
- **Season:** Spring has the highest cancellation rate (60.3%), Winter the lowest (45.4%) — noted as a business insight, though not included as a final model feature since it didn't improve model performance beyond what `Order_month` already captured

---

## 🖥️ Interactive Prediction App

A Streamlit web app (`cancellation_predictor_app.py`) is included so anyone can test the model without writing code:

```bash
pip install streamlit
streamlit run cancellation_predictor_app.py
```

Enter order details through a simple form (dropdowns for category/payment method, number inputs for price/quantity) and get an instant prediction with confidence score — no manual encoding required.

---

## 🛠️ Tech Stack
- **Python**, **pandas**, **scikit-learn**, **XGBoost**
- **Streamlit** for the interactive app

---

## 📌 Business Recommendation

Given Online Transfer's outsized role in predicting cancellations, the highest-leverage next step for the business is investigating **payment verification reliability** for Online Transfer orders — reducing friction there could meaningfully lower overall cancellation rates, more so than adjustments to pricing or category strategy alone.
