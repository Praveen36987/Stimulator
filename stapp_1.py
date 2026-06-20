import streamlit as st
import pandas as pd, numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score


# ============================ ENGINE ============================
class BaseSimulator(ABC):
    @abstractmethod
    def run(self, **levers): ...
    @abstractmethod
    def summary(self) -> str: ...

@dataclass
class PricingResult:
    price_change: float; units: float; revenue: float; profit: float; margin_pct: float

class PricingSimulator(BaseSimulator):
    def __init__(self, data, elasticity=-1.5):
        self._data = data; self.elasticity = elasticity
    def _economics(self):
        bp = self._data["final_price"].to_numpy(float)
        bq = self._data["sold_quantity"].to_numpy(float)
        cost = bp * (1 - self._data["profit_margin_pct"].to_numpy(float)/100)
        return bp, bq, cost
    def run(self, price_change=0.0):
        bp, bq, cost = self._economics(); r = 1 - price_change/100
        price, qty = bp*r, bq*(r**self.elasticity)
        rev = float((price*qty).sum()); prof = float(((price-cost)*qty).sum())
        return PricingResult(price_change, float(qty.sum()), rev, prof, prof/rev*100 if rev>0 else 0.0)
    def sweep(self, lo=-25, hi=50):
        return pd.DataFrame([asdict(self.run(p)) for p in range(lo, hi+1)])
    def optimal(self):
        s = self.sweep(); return PricingResult(**s.loc[s["profit"].idxmax()].to_dict())
    def summary(self):
        b, o = self.run(0), self.optimal()
        return f"[Pricing] elasticity {self.elasticity:+.1f} | today Rs{b.profit:,.0f} -> max at {o.price_change:+.0f}% (Rs{o.profit:,.0f})"

class BaseMLModel(ABC):
    def __init__(self, data):
        self._data = data; self._pipe = None; self.metrics = {}
    @abstractmethod
    def _columns(self): ...
    @abstractmethod
    def _estimator(self): ...
    @abstractmethod
    def _score(self, yt, yp): ...
    @abstractmethod
    def summary(self): ...
    def _make_pipeline(self, num, cat):
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
        return Pipeline([("pre", pre), ("model", self._estimator())])
    def fit(self):
        num, cat, target = self._columns()
        X, y = self._data[num+cat], self._data[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        self._pipe = self._make_pipeline(num, cat).fit(Xtr, ytr)
        self.metrics = self._score(yte, self._pipe.predict(Xte)); self._Xte, self._yte = Xte, yte
        return self

class DeliveryDelayModel(BaseMLModel):
    def _columns(self):
        self._data = self._data.assign(is_delayed=(self._data["delivery_status"]=="Delayed").astype(int))
        return ["delivery_time_min"], [], "is_delayed"
    def _estimator(self): return LogisticRegression(max_iter=1000)
    def _score(self, yt, yp): return {"accuracy": accuracy_score(yt, yp)}
    def fit(self):
        super().fit(); self.metrics["roc_auc"] = roc_auc_score(self._yte, self._pipe.predict_proba(self._Xte)[:,1]); return self
    def predict(self, t):
        return float(self._pipe.predict_proba(pd.DataFrame({"delivery_time_min":[float(t)]}))[0,1])
    def summary(self):
        return f"[Delivery] LogReg | acc {self.metrics['accuracy']:.3f}, AUC {self.metrics['roc_auc']:.3f} | P(delay@30)={self.predict(30)*100:.0f}%"

class SalesPredictor(BaseMLModel):
    NUM = ["demand_index","price","final_price","discount_pct","rating","num_reviews",
           "delivery_time_min","profit_margin_pct","weight_g","shelf_life_days","stock","reorder_level"]
    CAT = ["category","brand","city","seller","packaging_type","is_organic","offer_type"]
    def __init__(self, data, estimator=None):
        super().__init__(data); self._chosen = estimator if estimator is not None else GradientBoostingRegressor(random_state=42)
    def _columns(self): return self.NUM, self.CAT, "sold_quantity"
    def _estimator(self): return self._chosen
    def _score(self, yt, yp): return {"r2": r2_score(yt, yp), "mae": mean_absolute_error(yt, yp)}
    def compare_models(self):
        cands = {"LinearRegression": LinearRegression(),
                 "RandomForest": RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1),
                 "GradientBoosting": GradientBoostingRegressor(random_state=42)}
        rows = [{"model": n, **SalesPredictor(self._data, e).fit().metrics} for n, e in cands.items()]
        return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    def ablation(self):
        num_wo = [c for c in self.NUM if c != "demand_index"]
        X, y = self._data[num_wo+self.CAT], self._data["sold_quantity"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), self.CAT)], remainder="passthrough")
        pipe = Pipeline([("pre", pre), ("m", GradientBoostingRegressor(random_state=42))]).fit(Xtr, ytr)
        return {"with": round(self.metrics["r2"],3), "without": round(r2_score(yte, pipe.predict(Xte)),3)}
    def feature_importance(self, top=8):
        names = self._pipe.named_steps["pre"].get_feature_names_out(); imps = self._pipe.named_steps["model"].feature_importances_
        agg = {}
        for n, i in zip(names, imps):
            raw = n.split("__",1)[1] if "__" in n else n
            base = raw if raw in self.NUM else next((c for c in self.CAT if raw.startswith(c+"_")), raw)
            agg[base] = agg.get(base, 0.0) + i
        return pd.DataFrame({"feature": list(agg), "importance": list(agg.values())}).sort_values("importance", ascending=False).head(top).reset_index(drop=True)
    def predict(self, **overrides):
        d = {c: float(self._data[c].median()) for c in self.NUM}
        for c in self.CAT: d[c] = self._data[c].mode().iloc[0]
        d.update(overrides)
        return float(self._pipe.predict(pd.DataFrame([d])[self.NUM+self.CAT])[0])
    def summary(self):
        return f"[Sales] {type(self._chosen).__name__} | R2 {self.metrics['r2']:.3f}, MAE {self.metrics['mae']:.0f}"

class ProductSegmenter:
    FEATURES = ["final_price","rating","num_reviews","sold_quantity","demand_index","profit_margin_pct"]
    def __init__(self, data, k=4):
        self._data = data; self.k = k; self._pipe = None
    def fit(self):
        self._pipe = Pipeline([("scale", StandardScaler()), ("km", KMeans(n_clusters=self.k, n_init=10, random_state=42))]).fit(self._data[self.FEATURES])
        self._labels = self._pipe.named_steps["km"].labels_; return self
    def assign(self):
        d = self._data[self.FEATURES].copy(); d["segment"] = self._labels.astype(str); return d
    def profile(self):
        d = self._data[self.FEATURES].copy(); d["segment"] = self._labels
        return d.groupby("segment").mean().round(1)
    def summary(self):
        return f"[Segments] KMeans k={self.k} | sizes {pd.Series(self._labels).value_counts().sort_index().to_dict()}"

class BlinkitSimulator:
    def __init__(self, data):
        self._df = data.copy(); self._df["offer_type"] = self._df["offer_type"].fillna("None")
        self.delivery = DeliveryDelayModel(self._df).fit()
        self.sales = SalesPredictor(self._df).fit()
        self.segments = ProductSegmenter(self._df).fit()
    def pricing(self, category=None, elasticity=-1.5):
        data = self._df[self._df["category"]==category] if category else self._df
        return PricingSimulator(data, elasticity)


# ============================ STREAMLIT UI ============================
st.set_page_config(page_title="Blinkit Simulator", page_icon="🛒", layout="wide")

@st.cache_data
def load_data():
    # Reads the CSV from the same folder as this script (place blinkit_dataset.csv in the repo root)
    df = pd.read_csv("blinkit_dataset.csv")
    df["offer_type"] = df["offer_type"].fillna("None")
    return df

@st.cache_resource
def build_sim(_df):
    return BlinkitSimulator(_df)

df = load_data()
sim = build_sim(df)

st.title("🛒 Blinkit Quick-Commerce Simulator")
st.caption("Pull a lever, watch the outcome change. Powered by an OOP engine + ML models.")

tab1, tab2, tab3, tab4 = st.tabs(["💰 Pricing", "🚚 Delivery", "📈 Sales", "🧩 Segments"])

with tab1:
    c = st.columns(3)
    cat = c[0].selectbox("Category", ["All"] + sorted(df["category"].unique()))
    elasticity = c[1].slider("Elasticity (assumption)", -8.0, -0.2, -1.5, 0.1)
    change = c[2].slider("Price change %  (− raise · discount +)", -25, 50, 0)
    pricer = sim.pricing(None if cat == "All" else cat, elasticity)
    base, now, opt = pricer.run(0), pricer.run(change), pricer.optimal()
    k = st.columns(4)
    k[0].metric("Units", f"{now.units:,.0f}", f"{(now.units-base.units)/base.units*100:+.1f}%")
    k[1].metric("Revenue", f"₹{now.revenue/1e7:.2f} Cr", f"{(now.revenue-base.revenue)/base.revenue*100:+.1f}%")
    k[2].metric("Profit", f"₹{now.profit/1e7:.2f} Cr", f"{(now.profit-base.profit)/base.profit*100:+.1f}%")
    k[3].metric("Margin", f"{now.margin_pct:.1f}%", f"{now.margin_pct-base.margin_pct:+.1f} pts")
    st.info(f"💡 Profit-maximising move: **{opt.price_change:+.0f}%** price change → ₹{opt.profit/1e7:.2f} Cr profit.")
    st.line_chart(pricer.sweep().set_index("price_change")["profit"])

with tab2:
    t = st.slider("Delivery time (minutes)", 10, 56, 27)
    p = sim.delivery.predict(t)
    cc = st.columns(2)
    cc[0].metric("Predicted delay risk", f"{p*100:.1f}%")
    cc[1].metric("Model quality (AUC)", f"{sim.delivery.metrics['roc_auc']:.3f}")
    st.progress(min(int(p*100), 100), text=f"{p*100:.0f}% chance of delay")
    st.line_chart(pd.DataFrame({"min": range(10,57), "delay %": [sim.delivery.predict(x)*100 for x in range(10,57)]}).set_index("min"))

with tab3:
    left, right = st.columns(2)

    with left:
        d = st.slider("Demand index", 0, 100, 50)
        r = st.slider("Rating", 2.5, 5.0, 4.2, 0.1)
        pr = st.slider("Price (₹)", 10, 1000, 250, 10)
        scat = st.selectbox("Category", sorted(df["category"].unique()), key="scat")
        pred = sim.sales.predict(demand_index=d, rating=r, price=pr, category=scat)

        st.metric("Predicted units sold", f"{pred:,.0f}")
        if pred > 400:
            st.balloons()

    with right:
        st.bar_chart(sim.sales.feature_importance(8).set_index("feature")["importance"])
        ab = sim.sales.ablation()
        st.warning(f"With `demand_index` R²={ab['with']}, without it R²={ab['without']} → sales ride on the demand index.")
        st.dataframe(sim.sales.compare_models(), use_container_width=True, hide_index=True)

with tab4:
    st.caption(sim.segments.summary())
    st.scatter_chart(sim.segments.assign(), x="demand_index", y="sold_quantity", color="segment")
    st.dataframe(sim.segments.profile(), use_container_width=True)
