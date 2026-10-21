"""Portfolio analytics: weights, concentration, diversification and risk metrics."""

import math
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"symbol", "shares"}
CONCENTRATION_LIMIT = 0.25  # flag any single holding above 25% of the portfolio
TRADING_DAYS = 252
# Annualized volatility bands; a broad US stock index has historically run ~15-20%.
RISK_LEVELS = ((0.08, "low"), (0.18, "moderate"), (math.inf, "high"))

# Educational follow-ups shown next to each flag: what to learn, never what to buy or sell.
CONSIDER_CONCENTRATION = (
    "Read up on position sizing: when one holding is a large share of the portfolio, "
    "news about that one company moves the whole portfolio."
)
CONSIDER_FEW_HOLDINGS = (
    "Learn how broad index funds work: one fund can hold hundreds of companies, "
    "which spreads company-specific risk."
)
CONSIDER_ONE_ASSET_CLASS = (
    "Learn about asset allocation: bonds and cash usually swing less than stocks, "
    "and the usual mix depends on how soon the money is needed."
)


def load_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an uploaded CSV: lowercase columns, uppercase symbols, validate."""
    df = df.rename(columns=str.lower)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Portfolio CSV has no rows")
    if df["symbol"].isna().any() or df["shares"].isna().any():
        raise ValueError("Every row needs a symbol and a share count")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="raise")
    if (df["shares"] < 0).any():
        raise ValueError("Share counts can't be negative")
    if "asset_class" not in df.columns:
        df["asset_class"] = "equity"
    df["asset_class"] = df["asset_class"].fillna("equity").astype(str).str.lower().str.strip()
    # The same ticker held in two accounts is one position.
    return df.groupby("symbol", as_index=False, sort=False).agg(
        shares=("shares", "sum"), asset_class=("asset_class", "first")
    )


def analyze(holdings: pd.DataFrame, prices: dict[str, float]) -> dict[str, Any]:
    """Compute value, weights, and risk flags; unpriced holdings are reported, not guessed."""
    df = holdings.copy()
    df["price"] = df["symbol"].map(prices)
    unpriced = df.loc[df["price"].isna(), "symbol"].tolist()
    df = df.dropna(subset=["price"])
    df["value"] = df["shares"] * df["price"]
    total = float(df["value"].sum())
    df["weight"] = df["value"] / total if total else 0.0

    # Herfindahl-Hirschman index: 1/N for equal weights, 1.0 for a single holding.
    hhi = float((df["weight"] ** 2).sum())
    flags = [
        f"{row.symbol} is {row.weight:.0%} of the portfolio (above {CONCENTRATION_LIMIT:.0%})"
        for row in df.itertuples()
        if row.weight > CONCENTRATION_LIMIT
    ]
    considerations = [CONSIDER_CONCENTRATION] if flags else []
    if len(df) < 5:
        count = f"{len(df)} priced holding{'s' if len(df) != 1 else ''}"
        flags.append(f"Only {count} — limited diversification")
        considerations.append(CONSIDER_FEW_HOLDINGS)
    if df["asset_class"].nunique() == 1:
        flags.append(f"All holdings are {df['asset_class'].iloc[0]} — no asset-class mix")
        considerations.append(CONSIDER_ONE_ASSET_CLASS)

    return {
        "total_value": round(total, 2),
        "holdings": df[["symbol", "shares", "price", "value", "weight", "asset_class"]]
        .round(4)
        .to_dict("records"),
        "by_asset_class": df.groupby("asset_class")["weight"].sum().round(4).to_dict(),
        "hhi": round(hhi, 4),
        "effective_holdings": round(1 / hhi, 1) if hhi else 0,
        "flags": flags,
        "considerations": considerations,
        "unpriced": unpriced,
    }


def risk_metrics(weights: dict[str, float], histories: dict[str, list[dict]]) -> dict[str, Any]:
    """Volatility, max drawdown and period return of the weighted holdings, from daily closes.

    Holdings without history are left out and the rest re-weighted; they are listed in
    `missing_history` rather than guessed. Returns {} when there is too little data.
    """
    closes = pd.DataFrame(
        {
            symbol: pd.Series({bar["date"]: bar["close"] for bar in bars})
            for symbol, bars in histories.items()
            if symbol in weights and bars
        }
    )
    closes = closes.sort_index().dropna()  # keep only dates every holding traded
    if closes.shape[0] < 3:
        return {}
    returns = closes.pct_change().dropna()
    w = pd.Series(weights)[closes.columns]
    if w.sum() <= 0:
        return {}
    daily = returns @ (w / w.sum())
    growth = (1 + daily).cumprod()
    volatility = float(daily.std() * math.sqrt(TRADING_DAYS))
    return {
        "start_date": str(closes.index[0]),
        "end_date": str(closes.index[-1]),
        "trading_days": len(returns),
        "volatility": round(volatility, 4),
        "risk_level": next(label for limit, label in RISK_LEVELS if volatility < limit),
        "max_drawdown": round(float((growth / growth.cummax() - 1).min()), 4),
        "period_return": round(float(growth.iloc[-1] - 1), 4),
        "holding_volatility": {
            s: round(float(returns[s].std() * math.sqrt(TRADING_DAYS)), 4) for s in closes.columns
        },
        # Value of $100 invested at the start, for charting.
        "growth_of_100": [
            {"date": str(day), "value": round(100 * float(v), 2)}
            for day, v in pd.concat([pd.Series([1.0], index=[closes.index[0]]), growth]).items()
        ],
        "missing_history": sorted(set(weights) - set(closes.columns)),
    }
