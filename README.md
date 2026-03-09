# 🍔 CSAO — Context-Aware Cart Super Add-On Recommendation System

### Zomato Hackathon | LightGBM-Powered Real-Time Food Recommendations

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Hackathon%20Submission-red)

---

## 📌 About the Project

On food delivery platforms like Zomato, most users add only a primary item (like Biryani or Pizza) to their cart and miss complementary items such as drinks, desserts, or sides. This results in:

- Lower **Average Order Value (AOV)**
- Missed **cross-selling opportunities**
- Incomplete **meal experiences**
- Poor **revenue per transaction**

The **Cart Super Add-On (CSAO)** system solves this by intelligently recommending relevant add-on items in real time based on what is already in the user's cart.

The recommendation problem is framed as a **binary classification task**:
- **Label = 1** → Item was actually ordered (positive sample)
- **Label = 0** → Item was in the menu but NOT ordered in the same session (negative sample)

---

## 🎯 Objectives

- Predict the most likely add-on items a user will add to the cart
- Make dynamic recommendations that update as the cart changes in real time
- Maximize business KPIs: **AOV** and **add-on acceptance rate**
- Ensure real-time inference within **200–300 ms**
- Scale to handle **millions of prediction requests per day**

---

## 📁 Repository Structure

```
CSAO-Recommendation-System/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── dataset_analysis.ipynb         # EDA & data exploration
│   └── CSAO_Recommendation_ML.ipynb   # Model training & evaluation
├── data/
│   └── README.md                      # Dataset download instructions
├── models/
│   ├── csao_lgbm_model.pkl            # Trained LightGBM model
│   └── csao_features.pkl              # Saved feature list
└── reports/
    └── CSAO_Report.pdf                # Full project report
```

---

## 📊 Dataset

**Source:** [ShaikhBorhanUddin/Zomato-Data-Analysis](https://github.com/ShaikhBorhanUddin/Zomato-Data-Analysis)

| Table | Key Columns | Used For |
|-------|-------------|----------|
| `food.csv` | f_id, veg_or_non_veg | Item dietary type features |
| `menu.csv` | f_id, r_id, cuisine, price | Price, cuisine & item frequency features |
| `orders.csv` | order_id, user_id, r_id, order_date | Basket construction, temporal features |
| `restaurant.csv` | r_id, name, address, rating, cost | Restaurant-level features |
| `users.csv` | user_id | User-level features |

### Key EDA Findings (from `dataset_analysis.ipynb`)

- **72.9% Veg vs 27.1% Non-Veg** — dietary match feature added to prevent irrelevant cross-category recommendations
- **Price clusters in ₹100–200 range** — price compatibility used as a key interaction feature
- **Most orders contain 1–2 items** — confirms strong opportunity for add-on recommendations
- **Order data includes USD entries** — currency conversion applied (1 USD = ₹82.5)
- **Day-of-week order trends** — temporal features (hour, day, month) engineered from `order_date`

---

## ⚙️ Methodology

### Pipeline Overview

```
Historical Orders
       ↓
Basket Construction  →  [Biryani, Raita, Coke] per order
       ↓
Pairwise Dataset Creation
  Positive (Y=1): A→B, A→C, B→A, B→C ...
  Negative (Y=0): Items from menu never ordered together
       ↓
Feature Engineering
  User Features + Item Features + Interaction Features + Temporal Features
       ↓
LightGBM Classifier Training (80/20 split)
       ↓
Ranking by Predicted Probability → Top-8 Add-On Recommendations
```

### Feature Engineering (4 Dimensions)

| Feature Category | Features |
|-----------------|----------|
| **User Features** | user_order_count, user_avg_spend, user_total_spend |
| **Restaurant Features** | rest_order_count, rest_avg_price |
| **Item Features** | price, veg_or_non_veg, item_popularity_rank, price_group |
| **Contextual / Temporal** | hour, day_of_week, month, price_diff, same_restaurant_flag |

### Why Pairwise Learning?

Rather than predicting the entire cart sequence, the problem is reformulated as:

> **Given:** Item A (already in cart) + Item B (candidate add-on)  
> **Predict:** Y = 1 if B was historically ordered together with A, else Y = 0  
> **Score:** P(B added | A in cart) → rank all candidates → show Top-N

This approach directly models co-purchase relationships, works well with tabular structured data, is scalable, and is consistent with production-friendly architectures.

---

## 🤖 Model

### LightGBM Classifier — Hyperparameters

```python
lgb_model = LGBMClassifier(
    n_estimators      = 300,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 31,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    class_weight      = 'balanced',
    random_state      = 42
)
```

### Why LightGBM over Deep Learning?

1. **Faster inference** — ~5–50ms, well within the 300ms real-time constraint
2. **Handles class imbalance** — `class_weight='balanced'` manages the natural imbalance (most menu items are never ordered)
3. **Works well with tabular data** — no need for complex sequence modeling
4. **Interpretable** — feature importance is explainable to business stakeholders
5. **Production-friendly** — easy to deploy and maintain

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUC-ROC | Overall classifier performance |
| Precision@K | Fraction of top-K recommendations that are relevant |
| Recall@K | Fraction of relevant items captured in top-K |
| NDCG@K | Ranking quality — rewards placing relevant items higher |

---

## 📓 Notebooks

### 1. `dataset_analysis.ipynb` — Exploratory Data Analysis

Covers EDA across all 5 dataset tables:

- **food.csv** — Veg/Non-veg distribution, null value check
- **menu.csv** — Price distribution by bin (₹0–100, ₹100–200, etc.), cuisine analysis, top 20 food items (f_id frequency), top 20 restaurants (r_id frequency)
- **orders.csv** — Order volume over time, day-of-week patterns, top 20 order dates, currency value check
- **restaurant.csv** — Missing value handling, restaurant name extraction from URL links
- **users.csv** — User data overview and null check

### 2. `CSAO_Recommendation_ML.ipynb` — Model Training (10 Steps)

| Step | Description |
|------|-------------|
| **Step 0** | Install & import libraries (LightGBM, scikit-learn, pandas, seaborn) |
| **Step 1** | Load data — auto-download from GitHub dataset source via wget |
| **Step 2** | Merge all 5 tables & standardise column names |
| **Step 3** | Feature engineering — user RFM, restaurant, item & temporal features |
| **Step 4** | Create target variable — positive & negative sampling |
| **Step 5** | Train LightGBM classifier with 80/20 stratified split |
| **Step 6** | Model evaluation — AUC, confusion matrix, Precision@K, Recall@K, NDCG@K, ROC curve |
| **Step 7** | Real-time recommendation simulator (`get_recommendations()` function) |
| **Step 8** | Business metrics estimation — acceptance rate, AOV lift |
| **Step 9** | Segment analysis — model performance by user segment |
| **Step 10** | Save model as `csao_lgbm_model.pkl` and `csao_features.pkl` |

---

## 🏗️ System Architecture

### 5-Layer Production Design

```
┌─────────────────────────────────────────────┐
│  1. Data Storage Layer                       │
│     Orders, Menu, Food, Users, Restaurants   │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  2. Candidate Generation Layer               │
│     Same restaurant + Co-purchased items     │
│     + Cuisine filters + Veg/Non-veg filter   │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  3. Feature Engineering Layer                │
│     Real-time feature builder for each       │
│     candidate item vs current cart state     │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  4. Ranking Model Layer                      │
│     LightGBM → P(add-on | cart state)        │
│     Inference < 300ms                        │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  5. Serving & API Layer                      │
│     Top-8 results → CSAO Rail                │
│     Updates on every cart change             │
└─────────────────────────────────────────────┘
```

### Cold Start Handling

| Scenario | Fallback Strategy |
|----------|------------------|
| New User | Show popular add-ons for chosen restaurant |
| New Restaurant | Use cuisine-level co-purchase patterns |
| New Item | Use item category similarity + price bucket rules |

---

## 💼 Expected Business Impact

| KPI | Expected Impact |
|-----|----------------|
| Average Order Value (AOV) | ↑ Increase via relevant complementary item suggestions |
| Add-On Acceptance Rate | ↑ Higher precision = more accepted recommendations |
| Cart-to-Order Ratio | ↑ Better cart completeness reduces abandonment |
| Recommendation Fatigue | ↓ Dietary & price filters reduce irrelevant suggestions |
| Cold Start Coverage | ✅ 100% coverage via fallback chain |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended — No Setup Needed)

| Notebook | Link |
|----------|------|
| EDA Notebook | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HsaqNjhSE_akLOapiRKfzaz7DPkM3cNT) |
| ML Model Notebook | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17gI9Tz3x8VXEUzVpAvnpjyTTXZ49-sQZ) |

### Option 2: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CSAO-Recommendation-System
cd CSAO-Recommendation-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (run inside notebooks or terminal)
wget https://github.com/ShaikhBorhanUddin/Zomato-Data-Analysis/archive/refs/heads/main.zip
unzip main.zip

# 4. Run EDA notebook
jupyter notebook notebooks/dataset_analysis.ipynb

# 5. Run ML model training
jupyter notebook notebooks/CSAO_Recommendation_ML.ipynb
```

---

## 🔑 Key Design Decisions

1. **LightGBM over Deep Learning** — faster inference, easier to explain, works well with tabular data
2. **Negative sampling strategy** — menu items not ordered in the same session, critical for a realistic training dataset
3. **Meal-time feature** — biryani recommendations differ at lunch vs midnight
4. **Item popularity rank** — ensures well-known popular items get a baseline boost
5. **`class_weight='balanced'`** — handles the natural class imbalance (most menu items are not ordered in any given session)
6. **Dietary similarity filter** — veg/non-veg match prevents irrelevant cross-category suggestions
7. **Price compatibility feature** — price difference between cart item and candidate add-on improves relevance

---

## 🔮 Next Steps (Post-Hackathon)

- Add **item-item co-occurrence** features (e.g., if biryani in cart → recommend salan)
- Explore **Two-Tower Neural Network** for better personalization at scale
- Run **A/B test** online vs baseline to measure real AOV lift
- Implement **online learning** to adapt to changing user behavior in real time

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 📂 Dataset | https://github.com/ShaikhBorhanUddin/Zomato-Data-Analysis |
| 📓 EDA Colab Notebook | https://colab.research.google.com/drive/1HsaqNjhSE_akLOapiRKfzaz7DPkM3cNT |
| 🤖 ML Model Colab Notebook | https://colab.research.google.com/drive/17gI9Tz3x8VXEUzVpAvnpjyTTXZ49-sQZ |
| 📄 Project Report | See `reports/CSAO_Report.pdf` |

---

*Built for the Zomato Hackathon. Powered by LightGBM + XGBoost.*
