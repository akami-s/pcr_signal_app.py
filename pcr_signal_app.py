# ------------------------------------------------------------
# 🚀 PCR Signal App v1.7
#    - 中央：色付きバッジ＋矢印、40/60日後 表（上昇率↑／下落率↓）
#    - 教育タブ：同じ列構成で統一、差行は平均リターン差のみ
# ------------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="PCR Signal App", layout="centered")

# ============================================================
# 設定（帯区分・期間）
# ============================================================
LIVE_THRESHOLDS = {
    "UltraHigh": 0.90,  # 上位90%以上
    "High":      0.75,  # 上位75%以上
    "UltraLow":  0.10,  # 下位10%以下
    "Low":       0.20,  # 下位20%以下
}
LIVE_HORIZONS = [40, 60]     # 中央は 40 / 60 固定
WARN_SAMPLE   = 30           # N<これで「参考程度」

# ============================================================
# データ読み込み（mtime キャッシュキー）
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(path: str = "master.csv", _mtime: float = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.drop_duplicates(subset="Date").sort_values("Date")
    df["PCR"]    = pd.to_numeric(df["PCR"], errors="coerce")
    df["Nikkei"] = pd.to_numeric(df["Nikkei"], errors="coerce")
    df.set_index("Date", inplace=True)
    return df.dropna(subset=["PCR", "Nikkei"])

# ============================================================
# 基本ユーティリティ
# ============================================================
def forward_returns(df: pd.DataFrame, mask: pd.Series, horizon: int) -> np.ndarray:
    idx = df.index[mask]
    rets = []
    for d in idx:
        fwd = d + BDay(horizon)
        if fwd in df.index:
            rets.append(df.loc[fwd, "Nikkei"] / df.loc[d, "Nikkei"] - 1)
    return np.asarray(rets)

def band_stats_single(df, *, hi_thr=None, lo_thr=None, horizon: int):
    """単一帯 (High or Low) の平均・勝率・件数を返す"""
    if hi_thr is not None:
        mask = df["PCR"] >= hi_thr
    elif lo_thr is not None:
        mask = df["PCR"] <= lo_thr
    else:
        raise ValueError("hi_thr か lo_thr を指定してください")
    rets = forward_returns(df, mask, horizon)
    if len(rets) == 0:
        return {"mean": np.nan, "win": np.nan, "N": 0}
    return {"mean": rets.mean(), "win": (rets > 0).mean(), "N": len(rets)}

# ============================================================
# 教育用 High/Low 比較統計
# ============================================================
@st.cache_data(show_spinner=False)
def calc_stats(df, hi_q, lo_q, horizon):
    hi_thr = df["PCR"].quantile(hi_q)
    lo_thr = df["PCR"].quantile(lo_q)
    hi_rets = forward_returns(df, df["PCR"] >= hi_thr, horizon)
    lo_rets = forward_returns(df, df["PCR"] <= lo_thr, horizon)
    if len(hi_rets) < 10 or len(lo_rets) < 10:
        return None
    diff = lo_rets.mean() - hi_rets.mean()
    p_val = stats.ttest_ind(lo_rets, hi_rets, equal_var=False).pvalue
    return {
        "hi_thr": hi_thr, "lo_thr": lo_thr,
        "hi_mean": hi_rets.mean(), "lo_mean": lo_rets.mean(),
        "hi_pos": (hi_rets > 0).mean(), "lo_pos": (lo_rets > 0).mean(),
        "hi_N": len(hi_rets), "lo_N": len(lo_rets),
        "diff": diff, "p": p_val,
    }

# ============================================================
# グリッド検索（教育用ランキング）
# ============================================================
HI_GRID = np.arange(0.70, 0.96, 0.05)
LO_GRID = np.arange(0.30, 0.04, -0.05)
H_GRID  = [20, 40, 60, 80, 120, 180]

@st.cache_data(show_spinner=True)
def grid_search(df):
    rows = []
    for hi_q, lo_q in zip(HI_GRID, LO_GRID):
        hi_thr = df["PCR"].quantile(hi_q)
        lo_thr = df["PCR"].quantile(lo_q)
        hi_mask = df["PCR"] >= hi_thr
        lo_mask = df["PCR"] <= lo_thr
        for h in H_GRID:
            hi_rets = forward_returns(df, hi_mask, h)
            lo_rets = forward_returns(df, lo_mask, h)
            if len(hi_rets) < 10 or len(lo_rets) < 10:
                continue
            diff  = lo_rets.mean() - hi_rets.mean()
            p_val = stats.ttest_ind(lo_rets, hi_rets, equal_var=False).pvalue
            pooled = np.r_[hi_rets, lo_rets]
            score  = abs(diff) * np.sqrt(len(pooled)) / pooled.std(ddof=1)
            rows.append({
                "High%": int(hi_q*100), "Low%": int(lo_q*100),
                "H": h, "Diff%": diff*100, "p": p_val, "Score": score,
                "High_N": len(hi_rets), "Low_N": len(lo_rets)
            })
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

# ============================================================
# 中央シグナル用統計を事前計算
# ============================================================
@st.cache_data(show_spinner=False)
def precompute_live_tables(df):
    q90 = df["PCR"].quantile(LIVE_THRESHOLDS["UltraHigh"])
    q75 = df["PCR"].quantile(LIVE_THRESHOLDS["High"])
    q10 = df["PCR"].quantile(LIVE_THRESHOLDS["UltraLow"])
    q20 = df["PCR"].quantile(LIVE_THRESHOLDS["Low"])
    out = {b: {} for b in LIVE_THRESHOLDS}
    for h in LIVE_HORIZONS:
        out["UltraHigh"][h] = band_stats_single(df, hi_thr=q90, horizon=h)
        out["High"][h]      = band_stats_single(df, hi_thr=q75, horizon=h)
        out["UltraLow"][h]  = band_stats_single(df, lo_thr=q10, horizon=h)
        out["Low"][h]       = band_stats_single(df, lo_thr=q20, horizon=h)
    out["_thresholds"] = {"q90": q90, "q75": q75, "q20": q20, "q10": q10}
    return out

def classify_band(pcr, thr):
    if pcr >= thr["q90"]: return "UltraHigh"
    if pcr >= thr["q75"]: return "High"
    if pcr <= thr["q10"]: return "UltraLow"
    if pcr <= thr["q20"]: return "Low"
    return "Middle"

# ラベル・色設定
B_SHORT = {
    "UltraHigh": "極端高 (≧90%)",
    "High":      "高 (≧75%)",
    "UltraLow":  "極端低 (≦10%)",
    "Low":       "低 (≦20%)",
    "Middle":    "中立",
}
B_BG = {"UltraHigh":"#FFCDD2","High":"#FFEBEE","UltraLow":"#C8E6C9","Low":"#E8F5E9","Middle":"#ECEFF1"}
B_TX = {"UltraHigh":"#B71C1C","High":"#C62828","UltraLow":"#1B5E20","Low":"#2E7D32","Middle":"#37474F"}
SIG = {
    "UltraHigh":("⬇ 下落リスク","#D32F2F"),
    "High":     ("⬇ 下落傾向"  ,"#D32F2F"),
    "UltraLow": ("⬆ 上昇期待(強)","#388E3C"),
    "Low":      ("⬆ 上昇期待"   ,"#388E3C"),
    "Middle":   ("― 中立"      ,"#757575"),
}

def badge_html(b):
    return (f"<span style='background:{B_BG[b]};color:{B_TX[b]};"
            f"padding:2px 8px;border-radius:6px;font-size:1.1em;font-weight:600;'>"
            f"{B_SHORT[b]}</span>")
def sig_html(b):
    arrow, color = SIG[b]
    return (f"<span style='font-size:2.5em;color:{color};'>{arrow}</span>"
            f"<span style='font-size:1.2em;margin-left:8px;color:{color};'>{SIG[b][0]}</span>")

# ============================================================
# -------------    アプリ本体    ------------------------------
# ============================================================
PATH = "master.csv"
mtime = os.path.getmtime(PATH) if os.path.exists(PATH) else 0
df = load_data(PATH, _mtime=mtime)
live_tbl = precompute_live_tables(df); thr = live_tbl["_thresholds"]

# サイドバー（教育用）
st.sidebar.title("⚙️ パラメータ設定（教育用）")
hi_pct = st.sidebar.slider("High Percentile", 70, 95, 75, step=5)
lo_pct = st.sidebar.slider("Low Percentile", 5, 30, 20, step=5)
horizon = st.sidebar.selectbox("Horizon (営業日後 / 教育用)", [20,40,60,80,120,180], index=1)
edu_stats = calc_stats(df, hi_pct/100, lo_pct/100, horizon)

# データ最終日
st.caption(f"データ最終日: {df.index[-1].date()} / PCR={df.iloc[-1]['PCR']:.2f}")

# 判定日 & PCR
st.header("📈 PCR シグナル判定")
date_choice = st.selectbox("判定日", options=df.index, index=len(df.index)-1,
                           format_func=lambda d: d.strftime("%Y-%m-%d"))
input_pcr = st.number_input("PCR を入力", value=float(df.loc[date_choice,"PCR"]), step=0.01)

pct_rank = stats.percentileofscore(df["PCR"], input_pcr, kind="weak")/100
band     = classify_band(input_pcr, thr)

# 中央表示
st.subheader("現在のPCR位置とシグナル")
st.markdown(f"<div style='font-size:2.2em;font-weight:bold;'>{pct_rank*100:.1f}%</div>", unsafe_allow_html=True)
st.markdown(badge_html(band), unsafe_allow_html=True)
st.markdown(sig_html(band), unsafe_allow_html=True)

# テーブル作成
def make_row(h, stats_d):
    mean = stats_d["mean"]; win = stats_d["win"]; N = stats_d["N"]
    if N==0 or np.isnan(mean):
        return {"期間":f"{h}日後","平均リターン":"—","上昇率↑":"—","下落率↓":"—","サンプル":N,"注記":"データなし"}
    note = "参考程度" if N<WARN_SAMPLE else ""
    return {"期間":f"{h}日後","平均リターン":f"{mean*100:.2f}%","上昇率↑":f"{win*100:.1f}%",
            "下落率↓":f"{(1-win)*100:.1f}%","サンプル":N,"注記":note}

rows=[]
if band in ("UltraHigh","High","UltraLow","Low"):
    for h in LIVE_HORIZONS:
        rows.append(make_row(h, live_tbl[band][h]))

if rows:
    st.table(pd.DataFrame(rows).set_index("期間"))

# ----------------- 教育タブ -----------------
with st.expander("📊 データ探索 & 最適化ランキング"):
    st.subheader("選択パラメータの統計（教育用）")
    if edu_stats is None:
        st.warning("サンプルが少なすぎます。")
    else:
        st.write(f"High閾値: PCR≥{edu_stats['hi_thr']:.2f} (上位{hi_pct}%)")
        st.write(f"Low閾値 : PCR≤{edu_stats['lo_thr']:.2f} (下位{lo_pct}%)")
        st.write(f"Horizon : {horizon}営業日後")

        edu_tbl = pd.DataFrame([
            {"帯":"High","平均リターン":edu_stats["hi_mean"]*100,
             "上昇率↑":edu_stats["hi_pos"]*100,"下落率↓":(1-edu_stats["hi_pos"])*100,
             "サンプル数":edu_stats["hi_N"]},
            {"帯":"Low","平均リターン":edu_stats["lo_mean"]*100,
             "上昇率↑":edu_stats["lo_pos"]*100,"下落率↓":(1-edu_stats["lo_pos"])*100,
             "サンプル数":edu_stats["lo_N"]},
            {"帯":"差 (Low-High)","平均リターン":edu_stats["diff"]*100,
             "上昇率↑":np.nan,"下落率↓":np.nan,
             "サンプル数":edu_stats["hi_N"]+edu_stats["lo_N"]},
        ])
        for col in ["平均リターン","上昇率↑","下落率↓"]:
            edu_tbl[col] = edu_tbl[col].map(lambda x:f"{x:.2f}%" if pd.notna(x) else "—")
        st.table(edu_tbl.set_index("帯"))
        st.write(f"p値 (平均差の有意性): {edu_stats['p']:.3g}")

    # ランキングボタン（省略部分は前バージョンと同じまま）
    if st.button("🔁 ランキングを計算する"):
        grid_df = grid_search(df)
        if grid_df.empty:
            st.warning("有効なサンプルがありません")
        else:
            fmt = grid_df.copy()
            fmt["Diff%"] = fmt["Diff%"].map(lambda x:f"{x:.2f}")
            fmt["Score"] = fmt["Score"].map(lambda x:f"{x:.2f}")
            fmt["p_fmt"] = fmt["p"].map(lambda x:"≤0.001" if x<0.001 else f"{x:.3f}")
            fmt = fmt.drop(columns=["p"]).rename(columns={"p_fmt":"p"})
            st.dataframe(fmt, use_container_width=True)

st.caption("※過去データに基づく統計であり、将来の結果を保証するものではありません。")
