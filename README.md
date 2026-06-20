# 🛒 Blinkit Quick-Commerce: Pricing, Delivery & Demand Analysis

An end-to-end analytics project on a Blinkit-style quick-commerce catalog. The goal is not just to describe the data, but to answer four business questions a category manager would actually ask: *How should we price? Where does delivery risk come from? What drives sales? And which products behave alike?*

---

## 📌 The Business Problem

Quick-commerce runs on thin margins and minutes-fast delivery promises. Three decisions make or break the unit economics:

1. **Pricing** — a discount that lifts volume can still destroy profit if the demand response is weak.
2. **Delivery reliability** — late deliveries erode trust, but tightening promises has a cost. We need to know *when* delay risk actually spikes.
3. **Demand planning** — if we can't predict how many units a product sells, we over-stock perishables and under-stock winners.

This project builds a decision-support engine that turns the raw catalog into answers for each of these.

---

## 🎯 Objectives

- Quantify how **price changes** affect units, revenue, and profit, and find the profit-maximizing move.
- Model **delivery delay risk** as a function of promised delivery time.
- Predict **units sold** and identify which product attributes drive demand.
- Group products into **behavioral segments** for differentiated strategy.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | Blinkit product catalog (`blinkit_dataset.csv`) |
| **Rows** | [fill in — e.g. 25,000 products] |
| **Key fields** | `final_price`, `sold_quantity`, `profit_margin_pct`, `demand_index`, `delivery_time_min`, `delivery_status`, `rating`, `category`, `brand`, `city` |
| **Target variables** | `sold_quantity` (sales), `delivery_status` (delay), price response (simulated) |

---

## 🛠️ Tools & Methods

- **Python** — pandas, NumPy for data handling
- **scikit-learn** — Logistic Regression, Gradient Boosting, Random Forest, Linear Regression, K-Means
- **OOP design** — an abstract `BaseSimulator` / `BaseMLModel` hierarchy so each analysis module is independent and reusable
- **Streamlit** — interactive front-end so a non-technical user can pull the levers themselves

**Four analytical modules:**

1. **Pricing Simulator** — uses a price-elasticity assumption to sweep price changes from −25% to +50% and locate the profit peak.
2. **Delivery Delay Model** — Logistic Regression on delivery time, evaluated with accuracy and ROC-AUC.
3. **Sales Predictor** — Gradient Boosting regression on 12 numeric + 7 categorical features, benchmarked against Random Forest and Linear Regression, with a feature-importance and ablation study.
4. **Product Segmenter** — K-Means (k=4) on price, rating, reviews, sales, demand, and margin.

---

## 🔍 Key Findings

> Replace each `[fill in]` with the numbers your own run produces. The *direction* of each finding below is what the analysis is built to reveal — confirm it against your output.

### 1. Pricing
- The profit-maximizing move is a **[fill in, e.g. +8%]** price change, lifting profit from **₹[fill in] Cr** to **₹[fill in] Cr**.
- Because demand is elastic (assumed elasticity ≈ −1.5), deep discounts grow *units* but can shrink *profit* — the sweep makes the trade-off visible rather than guessed.

### 2. Delivery
- Delay risk is low for fast promises but climbs sharply past **[fill in] minutes**, where predicted delay probability crosses **[fill in]%**.
- The model separates on-time vs. delayed deliveries with an **AUC of [fill in]** — [strong / moderate] discriminative power.

### 3. Sales Drivers
- **`demand_index` is the single dominant predictor of units sold.** The ablation test confirms it: R² drops from **[fill in]** to **[fill in]** when the feature is removed.
- Gradient Boosting was the best model (R² = **[fill in]**, MAE = **[fill in]** units), beating Random Forest and Linear Regression.
- After demand, the next most influential factors were **[fill in — e.g. price, rating, category]**.

### 4. Product Segments
- K-Means produced **4 distinct segments** with sizes **[fill in]**.
- The segments roughly map to: **[fill in — e.g. "high-demand low-margin staples", "premium niche products", etc.]**, each warranting a different pricing and stocking approach.

---

## 💡 Recommendations

> Anchor these to your actual findings above.

1. **Pricing:** Move category pricing toward the simulated profit peak rather than defaulting to discounts; volume gains don't always pay for themselves.
2. **Delivery:** Set delivery-time promises *below* the risk-spike threshold; beyond it, the chance of breaking the promise rises faster than the time saved is worth.
3. **Demand planning:** Treat `demand_index` as the primary stocking signal, especially for perishables where over-ordering is costly.
4. **Segmentation:** Stop treating the catalog as one block — apply segment-specific pricing and inventory rules.

---



---

## 👤 Author
Praveen — 


*Built to turn a raw product catalog into pricing, delivery, and demand decisions.*
