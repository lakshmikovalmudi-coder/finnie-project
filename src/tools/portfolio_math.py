"""Portfolio analytics: weights, concentration, and diversification flags."""

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"symbol", "shares"}
CONCENTRATION_LIMIT = 0.25  # flag any single holding above 25% of the portfolio


def load_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an uploaded CSV: lowercase columns, uppercase symbols, validate."""
    df = df.rename(columns=str.lower)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV missing columns: {sorted(missing)}")
    df["symbol"] = df["symbol"].str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="raise")
    if "asset_class" not in df.columns:
        df["asset_class"] = "equity"
    return df


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
    if len(df) < 5:
        flags.append(f"Only {len(df)} priced holdings — limited diversification")
    if df["asset_class"].nunique() == 1:
        flags.append(f"All holdings are {df['asset_class'].iloc[0]} — no asset-class mix")

    return {
        "total_value": round(total, 2),
        "holdings": df[["symbol", "shares", "price", "value", "weight", "asset_class"]]
        .round(4)
        .to_dict("records"),
        "by_asset_class": df.groupby("asset_class")["weight"].sum().round(4).to_dict(),
        "hhi": round(hhi, 4),
        "effective_holdings": round(1 / hhi, 1) if hhi else 0,
        "flags": flags,
        "unpriced": unpriced,
    }
