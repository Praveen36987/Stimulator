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

### 1. Pricing — sales barely respond to price, so margin is the lever
A crucial data check came first: in the real catalog, **price is almost completely uncorrelated with units sold** (`final_price` vs `sold_quantity` = 0.002; `discount_pct` = 0.010). Sales simply don't move with price here. That's *why* pricing is modeled as an assumption-driven elasticity simulator rather than learned from data.

Running the simulator on the full catalog (assumed elasticity −1.5), the profit-maximizing move is to **raise prices**, lifting modeled profit from **₹11.52 Cr to ₹17.35 Cr (+51%)**. The optimum sits at the very edge of the tested range, implying the model would push prices even higher — a direct consequence of that near-zero price sensitivity.

> **Honest caveat:** this result is driven by the elasticity *assumption*, not observed behavior. Before acting on it, the recommendation would be to run a real price test (A/B) to measure true elasticity.

### 2. Delivery — risk climbs with promised time, but time alone explains only part of it
The Logistic Regression hits **78.7% accuracy** with an **ROC-AUC of 0.687** — modest discriminative power. At a 30-minute promise, predicted delay risk is **~21%**, and it rises steadily as delivery time increases.

The takeaway: delivery time genuinely affects delay risk, but an AUC of 0.69 means it's only a *partial* explanation. A stronger model would add city, seller, and weight.

### 3. Sales Drivers — one feature carries almost the entire model
**Gradient Boosting was the best predictor of units sold (R² = 0.886, MAE = 35 units)**, narrowly beating Random Forest (R² = 0.881) and Linear Regression (R² = 0.860).

But the headline is the **ablation test**: remove `demand_index` and R² collapses from **0.886 to −0.003** — i.e. every other feature combined predicts essentially nothing. Feature importance confirms it: `demand_index` accounts for **97%** of the model's decisions, with `rating` a distant second (~2%). The raw correlation between `demand_index` and `sold_quantity` is **0.914**.


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
