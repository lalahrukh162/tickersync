"""
╔══════════════════════════════════════════════════════════╗
║         TICKER SIMILARITY ENGINE — FLASK BACKEND         ║
╚══════════════════════════════════════════════════════════╝

WHY Flask and not FastAPI or Django?
→ Flask: minimal, explicit, perfect for small APIs like this
→ FastAPI: better for large APIs, async, auto-docs (good for production later)
→ Django: full framework with ORM, admin, too heavy for this use case

ENDPOINTS:
  GET  /api/tickers      → list of all tickers + date range
  GET  /api/config       → all default settings + slider ranges
  POST /api/search       → correlation search + normalized chart data
  GET  /api/full-market  → vectorized all-pairs correlation scan
  GET  /                 → serves index.html
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
import yfinance as yf
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── APP SETUP ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

DATA_FOLDER = os.environ.get("DATA_FOLDER", r"C:\Users\NEW_WORK_FOLDER\ohlcv_data\ohlcv_data")

DEFAULTS = {
    "min_correlation": 0.60,
    "min_overlap":     100,
    "lookback_days":   None,
    "min_score":       0.65,   # composite score gate (0.5×corr + 0.3×stability + 0.2×vol_sim)
}

SLIDER_RANGES = {
    "min_correlation": {"min": 0.3,  "max": 0.99, "step": 0.01},
    "min_overlap":     {"min": 30,   "max": 500,  "step": 10},
    "min_score":       {"min": 0.40, "max": 0.90, "step": 0.01},
}

# ── DATA LOADING ───────────────────────────────────────────────────────────────

print(f"\n📂 Loading ticker data from '{DATA_FOLDER}'...")


def load_all_returns(data_folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all parquet files, build prices + returns matrices.
    Returns: (prices_df, returns_df)
    """
    folder = Path(data_folder)
    files  = list(folder.glob("*.parquet"))

    if not files:
        print(f"⚠️  No parquet files found in '{data_folder}'")
        return pd.DataFrame(), pd.DataFrame()

    close_prices = {}
    for file in files:
        ticker = file.stem
        try:
            df = pd.read_parquet(file)
            col = "Close" if "Close" in df.columns else "close"

            # ── Auto-detect date column ──────────────────────────────
            date_col = None
            for candidate in ["Date", "date", "Datetime", "datetime", "timestamp", "index"]:
                if candidate in df.columns:
                    date_col = candidate
                    break

            if date_col is None:
                # Maybe the index is already the date
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)
                else:
                    print(f"  ⚠️  Skipping {ticker}: No date column found. Columns: {list(df.columns)}")
                    continue
            else:
                df = df.set_index(date_col)
                df.index = pd.to_datetime(df.index)

            if col in df.columns:
                close_prices[ticker] = df[col]

        except Exception as e:
            print(f"  ⚠️  Skipping {ticker}: {e}")

    prices_df = pd.DataFrame(close_prices)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df.sort_index(inplace=True)

    # fill_method=None → don't forward-fill NaN gaps before computing returns
    # WHY? Filling would create fake 0% return days and distort correlation.
    returns_df = prices_df.pct_change(fill_method=None).dropna(how="all")

    print(f"✅ Loaded {len(prices_df.columns)} tickers | "
          f"{prices_df.index[0].date()} → {prices_df.index[-1].date()}\n")

    return prices_df, returns_df


# ── SPY LOADING ────────────────────────────────────────────────────────────────

def load_spy_returns(returns_df: pd.DataFrame) -> pd.Series:
    """
    Load SPY returns for beta adjustment.
    Tries the data folder first, falls back to yfinance.

    WHY SPY?
    → Most universally accepted US market proxy.
    → Alternatives: QQQ (Nasdaq-heavy), IWM (small-caps).

    FIX 1: .squeeze() — modern yfinance (≥0.2) returns MultiIndex DataFrame.
    → spy_raw["Close"] gives a single-column DataFrame, NOT a Series.
    → .squeeze() converts single-column DataFrame → Series safely.

    FIX 2: .tz_localize(None) — yfinance may return tz-aware DatetimeIndex.
    → Your parquet data is tz-naive.
    → Mixing tz-aware and tz-naive causes alignment failures.
    → Strip timezone info so both indexes are comparable.
    """
    if "SPY" in returns_df.columns:
        print("  ✅ SPY found in local data — using as market benchmark")
        return returns_df["SPY"]

    print("  🌐 SPY not in local data — fetching from Yahoo Finance...")
    try:
        spy_raw = yf.download("SPY", start="2000-01-01",
                              progress=False, auto_adjust=True)

        # FIX 1: squeeze() handles MultiIndex DataFrame from modern yfinance
        spy_returns = spy_raw["Close"].squeeze().pct_change(fill_method=None).dropna()

        # FIX 2: strip timezone so index aligns with tz-naive parquet data.
        # WHY conditional? → tz_localize(None) crashes if index is ALREADY tz-naive.
        # yfinance sometimes returns tz-aware, sometimes tz-naive depending on version.
        # Check first, strip only if needed — safe either way.
        idx = pd.to_datetime(spy_returns.index)
        spy_returns.index = idx.tz_localize(None) if idx.tz is not None else idx

        # Align to our data's date index
        spy_aligned = spy_returns.reindex(returns_df.index)
        print(f"  ✅ SPY loaded: {spy_aligned.notna().sum()} trading days")
        return spy_aligned

    except Exception as e:
        print(f"  ⚠️  Could not load SPY: {e} — beta adjustment disabled")
        return pd.Series(dtype=float)


# ── METADATA ───────────────────────────────────────────────────────────────────

METADATA_CACHE_FILE = "ticker_metadata_cache.json"
METADATA_CACHE_DAYS = 7


def fetch_single_ticker_meta(ticker: str) -> dict:
    """
    Fetch metadata with ETF detection and ticker format fallback.

    WHY try multiple formats?
    → Parquet filenames may use underscores (BRK_B) but Yahoo needs hyphens (BRK-B)
    → Trying both costs one extra API call at most, saves many "Unknown" sectors

    WHY detect ETFs separately?
    → ETFs have no GICS sector — that's expected, not a failure
    → Labeling them "ETF" + category is more informative than "Unknown"
    → Prevents them from being confused with stocks that truly have no data
    """
    # Try original ticker first, then hyphen variant if underscore present
    variants = [ticker]
    if "_" in ticker:
        variants.append(ticker.replace("_", "-"))

    for t in variants:
        try:
            info = yf.Ticker(t).info

            # Skip empty responses (delisted / not found)
            if not info or info.get("trailingPegRatio") is None and info.get("sector") is None and info.get("quoteType") is None:
                continue

            quote_type = info.get("quoteType", "")

            # ETF handling — use category instead of sector
            if quote_type in ("ETF", "MUTUALFUND"):
                return {
                    "ticker":     ticker,
                    "name":       info.get("longName") or info.get("shortName") or ticker,
                    "sector":     quote_type,                          # "ETF" or "MUTUALFUND"
                    "industry":   info.get("category", "Unknown"),    # e.g. "Large Growth"
                    "market_cap": info.get("totalAssets"),             # ETFs use totalAssets
                    "country":    info.get("country", "Unknown"),
                }

            # Regular stock
            sector = info.get("sector", "Unknown") or "Unknown"
            if sector:
                return {
                    "ticker":     ticker,
                    "name":       info.get("longName") or info.get("shortName") or ticker,
                    "sector":     sector,
                    "industry":   info.get("industry", "Unknown") or "Unknown",
                    "market_cap": info.get("marketCap"),
                    "country":    info.get("country", "Unknown"),
                }

        except Exception as e:
            print(f"  ⚠️  {t}: {e}")
            continue

    # All variants failed
    return {
        "ticker":     ticker,
        "name":       ticker,
        "sector":     "Unknown",
        "industry":   "Unknown",
        "market_cap": None,
        "country":    "Unknown",
    }

def load_metadata_cache(tickers: list) -> dict | None:
    cache_path = Path(METADATA_CACHE_FILE)
    if not cache_path.exists():
        print("  ℹ️  No metadata cache found — will fetch fresh.")
        return None

    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)

        cached_at = datetime.fromisoformat(cache.get("cached_at", "2000-01-01"))
        age_days  = (datetime.now() - cached_at).days
        if age_days > METADATA_CACHE_DAYS:
            print(f"  ℹ️  Cache is {age_days} days old → refreshing.")
            return None

        data    = cache.get("data", {})
        missing = [t for t in tickers if t not in data]
        if missing:
            print(f"  ℹ️  Cache missing {len(missing)} tickers → will fetch those.")
            return data

        print(f"  ✅ Metadata cache hit: {len(data)} tickers (age: {age_days} days)")
        return data

    except Exception as e:
        print(f"  ⚠️  Cache read failed: {e} → fetching fresh.")
        return None


def save_metadata_cache(metadata: dict):
    try:
        cache = {"cached_at": datetime.now().isoformat(), "data": metadata}
        with open(METADATA_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  💾 Metadata cache saved → '{METADATA_CACHE_FILE}'")
    except Exception as e:
        print(f"  ⚠️  Could not save cache: {e} (non-fatal)")


def fetch_all_metadata(tickers: list) -> dict:
    print(f"\n🏷️  Loading sector/industry metadata for {len(tickers)} tickers...")

    cached_data = load_metadata_cache(tickers)
    metadata    = dict(cached_data) if cached_data else {}
    missing     = [t for t in tickers if t not in metadata]

    if not missing:
        _print_sector_summary(metadata)
        return metadata

    print(f"  🌐 Fetching {len(missing)} tickers from Yahoo Finance (10 parallel workers)...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_single_ticker_meta, t): t for t in missing}
        done = 0
        for future in as_completed(future_to_ticker):
            result = future.result()
            metadata[result["ticker"]] = result
            done += 1
            if done % 25 == 0 or done == len(missing):
                print(f"    ... {done}/{len(missing)} fetched")

    save_metadata_cache(metadata)
    _print_sector_summary(metadata)
    return metadata


def _print_sector_summary(metadata: dict):
    sectors = Counter(v.get("sector", "Unknown") for v in metadata.values())
    print(f"  📊 Sector distribution ({len(metadata)} tickers):")
    for sector, count in sectors.most_common():
        print(f"     {sector:<35} {count:>3}  {'█' * count}")
    print()



# ── STARTUP: Load all data once ────────────────────────────────────────────────

PRICES_DF, RETURNS_DF = load_all_returns(DATA_FOLDER)
SPY_RETURNS           = load_spy_returns(RETURNS_DF)
TICKER_META           = fetch_all_metadata(RETURNS_DF.columns.tolist()) if not RETURNS_DF.empty else {}


# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

def suggest_threshold(trading_days: int) -> float:
    """
    Suggest a realistic min_correlation based on how much data the target has.
    More data → more stable estimates → can afford stricter threshold.
    """
    if trading_days < 400:   return 0.55
    elif trading_days < 750: return 0.60
    elif trading_days < 1500: return 0.65
    else:                    return 0.70


def compute_rolling_stability(series_a: pd.Series, series_b: pd.Series, window: int = 63) -> float:
    """
    Measures consistency of correlation over time.

    HOW: Compute rolling 63-day (≈1 quarter) correlation windows.
    → Low std of rolling values = stable relationship = high score
    → High std = erratic relationship = low score
    → Returns 0.5 (neutral) if insufficient data
    """
    if len(series_a) < window * 2:
        return 0.5

    rolling_corr = series_a.rolling(window).corr(series_b).dropna()

    if len(rolling_corr) < 4:
        return 0.5

    stability = float(1.0 - rolling_corr.std())
    return max(0.0, min(1.0, stability))


def compute_vol_similarity(vol_a: float, vol_b: float) -> float:
    """
    Measures how similar the annualized volatility of two tickers is.

    Formula: 1 - |vol_a - vol_b| / max(vol_a, vol_b)
    → Equal vols → 1.0 (perfect)
    → One 2× the other → 0.5
    → One 3× the other → 0.33
    """
    if max(vol_a, vol_b) == 0:
        return 1.0

    sim = 1.0 - abs(vol_a - vol_b) / max(vol_a, vol_b)
    return max(0.0, min(1.0, sim))


def compute_beta_adjusted_residuals(ticker_returns: pd.Series,
                                     spy_returns: pd.Series) -> pd.Series:
    """
    Strip out market influence and return what the stock did BEYOND its beta.

    residual(t) = ticker_return(t) - beta × spy_return(t)

    WHY residuals?
    → Raw correlation measures "do both follow the market?"
    → Residual correlation measures "do they share idiosyncratic movement?"
    → The latter is a stronger signal of genuine behavioral similarity.

    Beta formula: Cov(ticker, SPY) / Var(SPY) — identical to OLS regression slope.

    Returns original series unchanged if insufficient SPY overlap (< 60 days).
    This is graceful degradation — system keeps working without beta adjustment.
    """
    combined = pd.DataFrame({"ticker": ticker_returns, "spy": spy_returns}).dropna()

    if len(combined) < 60:
        return ticker_returns

    cov_matrix = combined.cov()
    beta       = cov_matrix.loc["ticker", "spy"] / cov_matrix.loc["spy", "spy"]
    residuals  = combined["ticker"] - beta * combined["spy"]

    return residuals


def classify_score(score: float) -> str:
    if score >= 0.75: return "strong"
    elif score >= 0.60: return "moderate"
    else: return "weak"


# ── API ENDPOINTS ──────────────────────────────────────────────────────────────

@app.route("/api/tickers")
def get_tickers():
    if RETURNS_DF.empty:
        return jsonify({"error": "No data loaded. Check DATA_FOLDER in app.py"}), 500

    return jsonify({
        "tickers":    sorted(RETURNS_DF.columns.tolist()),
        "count":      len(RETURNS_DF.columns),
        "date_range": {
            "start": PRICES_DF.index[0].strftime("%Y-%m-%d"),
            "end":   PRICES_DF.index[-1].strftime("%Y-%m-%d"),
        },
    })


@app.route("/api/config")
def get_config():
    return jsonify({
        "defaults":      DEFAULTS,
        "slider_ranges": SLIDER_RANGES,
    })


@app.route("/api/search", methods=["POST"])
def search():
    if RETURNS_DF.empty:
        return jsonify({"error": "No data loaded"}), 500

    body          = request.get_json()
    target_ticker = body.get("ticker", "").upper().strip()
    min_corr      = float(body.get("min_correlation", DEFAULTS["min_correlation"]))
    min_overlap   = int(body.get("min_overlap",       DEFAULTS["min_overlap"]))
    lookback_days = body.get("lookback_days")
    min_score     = float(body.get("min_score",       DEFAULTS["min_score"]))

    if not target_ticker:
        return jsonify({"error": "No ticker provided"}), 400

    if target_ticker not in RETURNS_DF.columns:
        return jsonify({"error": f"Ticker '{target_ticker}' not found in your data."}), 404

    # ── STEP 1: Trim to target's valid date range ──────────────────────────────
    target_series_full = RETURNS_DF[target_ticker]
    target_first_valid = target_series_full.first_valid_index()
    target_last_valid  = target_series_full.last_valid_index()

    returns = RETURNS_DF.loc[target_first_valid:target_last_valid]

    if lookback_days:
        returns = returns.iloc[-int(lookback_days):]

    target_returns = returns[target_ticker]
    trading_days   = int(target_returns.notna().sum())
    target_start   = target_returns.first_valid_index().strftime("%Y-%m-%d")
    target_end     = target_returns.last_valid_index().strftime("%Y-%m-%d")

    _tvol      = target_returns.dropna().std() * np.sqrt(252) * 100
    target_vol = round(float(_tvol), 2) if pd.notna(_tvol) else 0.0

    print(f"\n🔍 Searching: {target_ticker}")
    print(f"   Window: {target_start} → {target_end} ({trading_days} days)")
    print(f"   Vol: {target_vol:.1f}% | Min corr: {min_corr} | Lookback: {lookback_days or 'ALL'}")

    # ── FIX 2+3: Compute spy_window and target_resid ONCE before the loop ──────
    #
    # WHY outside the loop?
    # → spy_window: SPY_RETURNS.reindex(returns.index) produces the same result
    #   every iteration — computing it 160× inside the loop is pure waste.
    # → target_resid: target_returns never changes during the loop.
    #   Computing its residuals 160× is 160× redundant work.
    # → Moving both outside = same result, ~160× less computation.

    spy_window = SPY_RETURNS.reindex(returns.index)
    beta_enabled = bool(SPY_RETURNS.notna().sum() > 60)

    if beta_enabled:
        target_resid_full = compute_beta_adjusted_residuals(
            target_returns.dropna(),
            spy_window[target_returns.dropna().index]
        )
        print(f"   Beta adjustment: ✅ ENABLED (SPY overlap: {spy_window.notna().sum()} days)")
    else:
        target_resid_full = None
        print(f"   Beta adjustment: ⚠️  DISABLED (insufficient SPY data)")

    # ── SEARCH LOOP ────────────────────────────────────────────────────────────

    results   = []
    all_corrs = []

    for ticker in returns.columns:
        if ticker == target_ticker:
            continue

        other = returns[ticker]

        # Pairwise valid mask — only dates where BOTH tickers have real data
        valid   = target_returns.notna() & other.notna()
        overlap = int(valid.sum())

        if overlap < min_overlap:
            continue

        # ── Raw (Pearson) correlation — always computed ────────────────────────
        #
        # WHY compute raw even when beta adjustment is enabled?
        # → We need it for beta_gap = raw_corr - beta_adjusted_corr
        # → beta_gap tells you: "how much of the similarity vanishes after
        #   stripping market noise?"
        # → Low gap → genuine peers. High gap → both just follow SPY.

        raw_corr = float(target_returns[valid].corr(other[valid]))

        # ── Beta-adjusted correlation ──────────────────────────────────────────
        #
        # target_resid_full already computed above (once, not per-ticker).
        # We only need to compute other_resid here (unique per ticker).
        #
        # WHY correlate residuals and not raw returns?
        # → Raw correlation: "do both follow the market?"
        # → Residual correlation: "do they share movement BEYOND market noise?"
        # → Residual version is a stronger signal of genuine behavioral peers.

        if beta_enabled and target_resid_full is not None:
            other_resid = compute_beta_adjusted_residuals(
                other[valid],
                spy_window[other[valid].index]
            )
            common_idx = target_resid_full.index.intersection(other_resid.index)

            if len(common_idx) >= min_overlap:
                corr = float(target_resid_full[common_idx].corr(other_resid[common_idx]))
            else:
                # Residual alignment reduced overlap below minimum → use raw
                corr = raw_corr
        else:
            # SPY unavailable → graceful fallback to raw Pearson correlation
            corr = raw_corr

        if pd.isna(corr):
            continue

        all_corrs.append(corr)

        if corr < min_corr:
            continue

        # Overlap date range (for display)
        common_dates  = returns.index[valid]
        overlap_start = common_dates[0].strftime("%Y-%m-%d")
        overlap_end   = common_dates[-1].strftime("%Y-%m-%d")

        # Annualized volatility: daily_std × √252 × 100
        _vol      = other[valid].std() * np.sqrt(252) * 100
        other_vol = round(float(_vol), 2) if pd.notna(_vol) else 0.0

        # Rolling stability (consistency of correlation over time)
        stability = compute_rolling_stability(target_returns[valid], other[valid])

        # Volatility similarity
        vol_sim = compute_vol_similarity(target_vol, other_vol)

        # ── Composite score ────────────────────────────────────────────────────
        # 50% correlation + 30% stability + 20% vol similarity
        # All three components flow in naturally — no thresholds or penalties.
        # A ticker with high corr but low stability will simply score lower
        # than one with high corr AND high stability. The weights do the work.
        score = 0.5 * corr + 0.3 * stability + 0.2 * vol_sim
        score = round(score, 4)

        # Sector metadata + cross-sector detection
        meta_other    = TICKER_META.get(ticker, {})
        meta_target   = TICKER_META.get(target_ticker, {})
        other_sector  = meta_other.get("sector",  "Unknown")
        target_sector = meta_target.get("sector", "Unknown")

        is_cross_sector = (
            other_sector  != target_sector and
            other_sector  != "Unknown"     and
            target_sector != "Unknown"
        )

        # ── Beta Gap ───────────────────────────────────────────────────────────
        #
        # WHY: beta_gap = raw_corr - beta_adjusted_corr
        # → If the gap is large, the stocks' similarity is mostly because
        #   both rise/fall with the market (SPY beta). Strip it and they diverge.
        # → If the gap is small, their co-movement survives market stripping →
        #   they are genuine behavioral peers independent of macro noise.
        #
        # Interpretation:
        #   < 0.05  → GENUINE  (real similarity, not market-driven)
        #   0.05–0.15 → MIXED  (partial genuine, partial market)
        #   > 0.15  → MARKET-DRIVEN (just both following SPY)

        if beta_enabled:
            beta_gap = round(float(raw_corr - corr), 4)
        else:
            beta_gap = None   # Can't compute without SPY

        results.append({
            "ticker":              ticker,
            "score":               score,
            "score_label":         classify_score(score),
            "correlation":         round(corr, 4),
            "raw_correlation":     round(raw_corr, 4),
            "beta_gap":            beta_gap,
            "stability":           round(stability, 4),
            "vol_similarity":      round(vol_sim, 4),
            "volatility":          other_vol,
            "overlap_days":        overlap,
            "overlap_start":       overlap_start,
            "overlap_end":         overlap_end,
            "name":                meta_other.get("name",     ticker),
            "sector":              other_sector,
            "industry":            meta_other.get("industry", "Unknown"),
            "market_cap":          meta_other.get("market_cap"),
            "country":             meta_other.get("country",  "Unknown"),
            "cross_sector":        is_cross_sector,
        })

    # ── Debug output ───────────────────────────────────────────────────────────
    top_corr_found = None
    if all_corrs:
        all_corrs.sort(reverse=True)
        top_corr_found = all_corrs[0]
        print(f"   Evaluated: {len(all_corrs)} tickers")
        print(f"   Top 5 corrs: {[round(c, 4) for c in all_corrs[:5]]}")
        print(f"   Range: {min(all_corrs):.4f} → {max(all_corrs):.4f} | Above {min_corr}: {len(results)}")
    else:
        print(f"   ⚠️  No tickers had >={min_overlap} overlapping days with {target_ticker}.")

    # Sort by composite score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Score gate — user-controlled from the dashboard slider
    # WHY after sorting? → Score must be fully computed first.
    # WHY not inside the loop? → Avoids computing stability/vol_sim for pairs
    #   we'd only discard — but actually we compute those anyway for the composite.
    #   Post-sort filter is cleaner and equally efficient.
    # A result must pass BOTH min_corr (correlation gate) AND min_score (quality gate).
    results = [r for r in results if r["score"] >= min_score]

    # ── Build chart data (per-pair, overlap-anchored, dual mode) ──────────────
    #
    # WHY per-pair instead of one shared blob?
    # → Old approach: send entire target history as one flat array, slice on
    #   frontend by overlap_start/overlap_end. Problem: the % base price was
    #   the ticker's all-time first price, so a ticker that existed for 3 years
    #   before the overlap would arrive at the overlap window already at +80%
    #   or -30%, making correlated pairs look completely different visually.
    # → New approach: for each matched pair, slice PRICES_DF to exactly the
    #   shared overlap window, then compute % / z-score from that window's
    #   Day 0. Both tickers always start at 0% (or 0σ) on the same date.
    #   This is what TradingView does when you compare two tickers.
    #
    # WHY send BOTH pct and zscore from the backend?
    # → The frontend has a toggle (% Return | Z-Score). If we only sent one
    #   mode, the user would have to re-fetch every time they switch. Sending
    #   both means the toggle is instant — just swap which dataset Chart.js uses.
    # → Data size is negligible: each pair's overlap is typically 200–800 rows
    #   × 2 columns × 2 modes = small.
    #
    # Z-SCORE formula: (value - mean) / std  over the overlap window.
    # → Removes both scale and level. Two tickers with identical shape but
    #   one moving $1 and other $100 will overlap perfectly on z-score.
    # → This is the standard quant visual for "do these move together?"
    # → Mean and std are computed per ticker over its own overlap window only.
    #
    # % RETURN formula: (price / price_at_overlap_start - 1) * 100
    # → Shows real gain/loss from the common starting point.
    # → Correlated tickers track each other because scale is real % terms.
    # → ffill() handles isolated NaN gaps (halted days) without breaking the line.

    chart_data_by_ticker: dict = {}

    for r in results:
        ticker       = r["ticker"]
        ov_start     = r["overlap_start"]   # already "YYYY-MM-DD" string
        ov_end       = r["overlap_end"]

        tickers_needed = [target_ticker, ticker]
        available      = [t for t in tickers_needed if t in PRICES_DF.columns]
        if len(available) < 2:
            continue

        # Slice to the exact overlap window for this pair
        pair_prices = PRICES_DF.loc[ov_start:ov_end, available].copy()

        # ── FIX: Double lookback REMOVED ──────────────────────────────────────
        # WHY removed? lookback_days was already applied to `returns` in STEP 1
        # (line ~500). overlap_start/overlap_end were derived from that already-
        # trimmed window. Applying lookback again here would cut the chart window
        # shorter than the correlation window → number ≠ graph.
        # The overlap window IS the correct window. No further slicing needed.

        if pair_prices.empty or len(pair_prices) < 2:
            continue

        # ffill isolated NaN gaps (halted days) — only within this window
        pair_prices = pair_prices.ffill()

        # ── % Return: anchor both to 0% at overlap window start ───────────────
        base = pair_prices.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else np.nan)
        pct  = ((pair_prices / base) - 1) * 100
        pct  = pct.round(4)

        # ── Z-Score: standardize each ticker over this overlap window ──────────
        # WHY dropna() before mean/std?
        # → A ticker may have a few NaN rows even after ffill (e.g. at the very
        #   start if it listed after the overlap started). Using those NaNs in
        #   mean/std would produce NaN for the whole column. dropna() per column
        #   ensures we compute stats only from real prices.
        mean = pair_prices.apply(lambda col: col.dropna().mean())
        std  = pair_prices.apply(lambda col: col.dropna().std())
        std  = std.replace(0, np.nan)   # avoid division by zero for flat lines
        zscore = ((pair_prices - mean) / std).round(4)

        # ── Adj Return: beta-stripped cumulative returns ───────────────────────
        #
        # WHY? adj corr score is computed on SPY-residuals, not raw prices.
        # So the "Adj Return" chart should also show SPY-stripped movement —
        # this way the chart visually matches the adj corr number shown in label.
        #
        # HOW:
        #   1. Get daily returns for the overlap window
        #   2. Strip SPY beta from each ticker → residuals
        #   3. Cumulative sum of residuals → cumulative "alpha" curve
        #   4. Multiply by 100 → % terms, anchored at 0 on Day 0
        #
        # WHY cumsum (not cumprod)?
        # → Residuals can be negative. cumprod of small residuals drifts badly.
        # → cumsum is standard for residual/alpha charts in quant finance.

        adj_return_df = pd.DataFrame(index=pair_prices.index)

        if beta_enabled:
            for t in available:
                raw_rets = RETURNS_DF.loc[ov_start:ov_end, t].copy()
                spy_win  = SPY_RETURNS.reindex(raw_rets.index)

                residuals = compute_beta_adjusted_residuals(
                    raw_rets.dropna(),
                    spy_win[raw_rets.dropna().index]
                )

                # Reindex back to full overlap window (fills gaps with NaN)
                residuals = residuals.reindex(pair_prices.index)

                # Cumulative sum → % alpha from Day 0
                adj_return_df[t] = (residuals.cumsum() * 100).round(4)
        else:
            # SPY unavailable → fall back to raw % return (same as pct)
            adj_return_df = pct.copy()

        # ── Format dates and serialize ─────────────────────────────────────────
        def to_records(df: pd.DataFrame) -> list:
            d = df.reset_index()
            dc = d.columns[0]
            d[dc] = pd.to_datetime(d[dc]).dt.strftime("%Y-%m-%d")
            d = d.rename(columns={dc: "date"})
            return json.loads(d.to_json(orient="records"))

        chart_data_by_ticker[ticker] = {
            "pct":        to_records(pct),
            "zscore":     to_records(zscore),
            "adj_return": to_records(adj_return_df),
        }

    # Legacy flat chart_data kept empty — frontend now uses chart_data_by_ticker
    chart_data = []

    suggested = suggest_threshold(trading_days)

    print(f"   Chart: {len(chart_data_by_ticker)} pairs built (per-pair, overlap-anchored, pct+zscore)")
    print(f"   Suggested threshold: {suggested}")

    # ── FIX: target_name, target_sector, cross_sector_count at TOP LEVEL ───────
    #
    # WHY here and not inside results.append()?
    # → cross_sector_count needs the COMPLETE results list to count correctly.
    #   Inside the loop, results is only partially built — count is always wrong.
    # → target_name/sector are the same value for every result — sending them
    #   once at the top level is cleaner and avoids 20× redundant repetition.

    return jsonify({
        "target":                target_ticker,
        "target_start":          target_start,
        "target_end":            target_end,
        "target_vol":            round(target_vol, 2),
        "trading_days":          trading_days,
        "results":               results,
        "total_found":           len(results),
        "chart_data":            chart_data,             # legacy empty list
        "chart_data_by_ticker":  chart_data_by_ticker,  # new: per-pair, dual-mode
        "suggested_min_corr":    suggested,
        "top_correlation_found": round(top_corr_found, 4) if top_corr_found else None,
        "beta_adjustment":       beta_enabled,
        # ── Target metadata (once at top level, not per result) ────────────────
        "target_name":           TICKER_META.get(target_ticker, {}).get("name",     target_ticker),
        "target_sector":         TICKER_META.get(target_ticker, {}).get("sector",   "Unknown"),
        "target_industry":       TICKER_META.get(target_ticker, {}).get("industry", "Unknown"),
        "target_market_cap":     TICKER_META.get(target_ticker, {}).get("market_cap"),
        # ── Computed AFTER full results list is built ──────────────────────────
        "cross_sector_count":    sum(1 for r in results if r.get("cross_sector")),
        "settings_used": {
            "min_correlation": min_corr,
            "min_overlap":     min_overlap,
            "lookback_days":   lookback_days,
            "min_score":       min_score,
        },
    })


# ── FULL MARKET SCAN ENDPOINT ─────────────────────────────────────────────────

@app.route("/api/full-market", methods=["GET"])
def full_market():
    """
    Compute pairwise correlations for ALL tickers at once.
    Returns grouped list: each ticker → its correlated peers above threshold.

    STEP 1 — Vectorized correlation matrix (fast, ~2-3s):
        pandas .corr() computes full N×N adj + raw matrices in one shot.

    STEP 2 — Score computation (per qualifying pair only):
        Only pairs that pass min_corr AND min_score gates get stability +
        vol_similarity computed. This avoids computing expensive rolling
        stability for the thousands of pairs that would be filtered anyway.
        Typical 160-ticker scan: ~500-2000 qualifying pairs → still fast.

    WHY not reuse /api/search 160×?
        That would re-compute SPY residuals 160×, re-slice data 160×,
        re-sort 160× — roughly 160× slower for the same result.

    Query params:
        min_correlation  (float, default 0.60) — minimum adj correlation
        min_score        (float, default 0.65) — minimum composite score
        min_overlap      (int,   default 100)  — minimum shared trading days
        lookback_days    (int,   optional)     — limit history window
    """
    if RETURNS_DF.empty:
        return jsonify({"error": "No data loaded"}), 500

    min_corr      = float(request.args.get("min_correlation", 0.60))
    min_score     = float(request.args.get("min_score",       0.65))
    min_overlap   = int(request.args.get("min_overlap",       100))
    lookback_days = request.args.get("lookback_days")
    if lookback_days:
        lookback_days = int(lookback_days)

    print(f"\n🌐 Full Market Scan | min_corr={min_corr} | min_score={min_score} | lookback={lookback_days or 'ALL'}")

    # ── Trim to lookback window ────────────────────────────────────────────────
    returns = RETURNS_DF.copy()
    if lookback_days:
        returns = returns.iloc[-lookback_days:]

    tickers = returns.columns.tolist()

    # ── STEP 1A: Beta-adjust all returns ──────────────────────────────────────
    beta_enabled = bool(SPY_RETURNS.notna().sum() > 60)
    spy_window   = SPY_RETURNS.reindex(returns.index)

    if beta_enabled:
        print(f"   Beta adjustment: ✅ ENABLED")
        residuals = {}
        for t in tickers:
            s = returns[t].dropna()
            if len(s) < 60:
                residuals[t] = returns[t]
                continue
            residuals[t] = compute_beta_adjusted_residuals(
                s, spy_window[s.index]
            ).reindex(returns.index)
        work_df = pd.DataFrame(residuals)
    else:
        print(f"   Beta adjustment: ⚠️  DISABLED")
        work_df = returns.copy()

    # ── STEP 1B: Full correlation matrices (vectorized) ───────────────────────
    print(f"   Computing {len(tickers)}×{len(tickers)} correlation matrix...")
    corr_matrix = work_df.corr(min_periods=min_overlap)   # adj corr
    raw_matrix  = returns.corr(min_periods=min_overlap)    # raw corr
    print(f"   Matrix done. Computing vol per ticker...")

    # ── STEP 1C: Annualized vol per ticker (vectorized) ───────────────────────
    # WHY precompute? vol_similarity needs vol_a and vol_b per pair.
    # Computing it once per ticker (not per pair) saves N× work.
    vols = (returns.std() * np.sqrt(252) * 100).to_dict()

    print(f"   Filtering pairs above corr={min_corr}, score={min_score}...")

    # ── STEP 2: Per-pair score computation (only for qualifying pairs) ─────────
    #
    # WHY not vectorize stability?
    # → Rolling correlation requires per-pair series alignment — no matrix form.
    # → But we only compute it for pairs already above min_corr, which is a
    #   small fraction of the 160×160=25,600 possible pairs.
    # → Typical result: 200-800 pairs above threshold → fast enough.

    grouped     = {}
    total_pairs = 0
    skipped_score = 0

    for ticker in tickers:
        if ticker not in corr_matrix.columns:
            continue

        col     = corr_matrix[ticker].drop(labels=[ticker], errors="ignore")
        raw_col = raw_matrix[ticker].drop(labels=[ticker],  errors="ignore")

        # First gate: adj corr threshold (fast, vectorized already done)
        above = col[col >= min_corr].dropna().sort_values(ascending=False)
        if above.empty:
            continue

        ticker_returns = returns[ticker]
        ticker_vol     = vols.get(ticker, 0.0)

        peers = []
        for peer, adj_c in above.items():
            raw_c = raw_col.get(peer, float("nan"))

            # ── Compute score components ───────────────────────────────────────
            peer_returns = returns[peer]
            valid        = ticker_returns.notna() & peer_returns.notna()
            overlap      = int(valid.sum())

            if overlap < min_overlap:
                continue

            # Stability: rolling 63-day correlation consistency
            stability = compute_rolling_stability(
                ticker_returns[valid], peer_returns[valid]
            )

            # Vol similarity
            peer_vol = vols.get(peer, 0.0)
            vol_sim  = compute_vol_similarity(ticker_vol, peer_vol)

            # Composite score — same formula as /api/search
            # 50% adj corr + 30% stability + 20% vol similarity
            score = round(0.5 * float(adj_c) + 0.3 * stability + 0.2 * vol_sim, 4)

            # Second gate: score threshold
            if score < min_score:
                skipped_score += 1
                continue

            beta_gap = round(float(raw_c - adj_c), 4) if not pd.isna(raw_c) else None
            score_label = classify_score(score)

            meta = TICKER_META.get(peer, {})
            peers.append({
                "ticker":      peer,
                "name":        meta.get("name",   peer),
                "sector":      meta.get("sector", "Unknown"),
                "adj_corr":    round(float(adj_c), 4),
                "raw_corr":    round(float(raw_c), 4) if not pd.isna(raw_c) else None,
                "beta_gap":    beta_gap,
                "score":       score,
                "score_label": score_label,
                "stability":   round(stability, 4),
                "vol_sim":     round(vol_sim, 4),
            })

        if not peers:
            continue

        # Sort peers by score descending (not just corr)
        peers.sort(key=lambda x: x["score"], reverse=True)
        total_pairs += len(peers)

        meta_t = TICKER_META.get(ticker, {})
        grouped[ticker] = {
            "ticker":     ticker,
            "name":       meta_t.get("name",     ticker),
            "sector":     meta_t.get("sector",   "Unknown"),
            "industry":   meta_t.get("industry", "Unknown"),
            "peers":      peers,
            "peer_count": len(peers),
            "best_score": peers[0]["score"] if peers else 0,
        }

    # Sort grouped: tickers with highest best_score first
    sorted_grouped = dict(
        sorted(grouped.items(), key=lambda x: x[1]["best_score"], reverse=True)
    )

    print(f"   ✅ {len(sorted_grouped)} tickers have peers | {total_pairs} total pairs")
    print(f"   Skipped by score gate: {skipped_score} pairs")

    return jsonify({
        "tickers_scanned":    len(tickers),
        "tickers_with_peers": len(sorted_grouped),
        "total_pairs":        total_pairs,
        "min_correlation":    min_corr,
        "min_score":          min_score,
        "lookback_days":      lookback_days,
        "beta_adjustment":    beta_enabled,
        "results":            sorted_grouped,
    })


# ── SERVE FRONTEND ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── START SERVER ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        debug=False,        # MUST be False when sharing — debug=True is a security hole
        host="0.0.0.0",     # Listen on all interfaces (local + network)
        port=5000,          # Change this if port 5000 is taken on your machine
        threaded=True,      # Handle multiple users at once (one thread per request)
                            # WHY threaded? Without it, Flask processes one request
                            # at a time — if one user triggers a full market scan
                            # (slow), every other user is frozen waiting.
    )