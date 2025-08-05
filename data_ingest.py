import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────
TICKERS  = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]
START    = "2005-01-01"
END      = "2025-01-01"  # exclusive end
INTERVAL = "1d"            # "1d", "1wk", etc.

# ─────────────────────────────────────────────────────────────────────
# Data Ingestion
# ─────────────────────────────────────────────────────────────────────
def fetch_price_series(tickers, start, end, interval="1d") -> pd.DataFrame:
    print("🔹 Downloading prices …")
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        progress=False,
        threads=False,
        auto_adjust=False
    )
    if df.empty:
        raise RuntimeError("No data downloaded — check tickers or dates.")
    if len(tickers) == 1:
        return df["Adj Close"].to_frame(tickers[0])
    prices = pd.concat({t: df[t]["Adj Close"] for t in tickers}, axis=1)
    return prices.dropna(how="all")

# ─────────────────────────────────────────────────────────────────────
# Signal Computation
# ─────────────────────────────────────────────────────────────────────
def compute_basket_zscores(price_df: pd.DataFrame) -> pd.DataFrame:
    clean = price_df.ffill().dropna(how="all")
    basket_mean = clean.mean(axis=1)
    basket_std  = clean.std(axis=1)
    zscores = clean.sub(basket_mean, axis=0).div(basket_std, axis=0)
    return zscores

# ─────────────────────────────────────────────────────────────────────
# Position Generation and Backtest
# ─────────────────────────────────────────────────────────────────────
def generate_positions(zscores: pd.DataFrame, entry_thresh: float = -1.0, exit_thresh: float = 0.0) -> pd.DataFrame:
    pos = (zscores < entry_thresh).astype(int)
    pos[zscores > exit_thresh] = 0
    return pos.ffill().fillna(0)

def backtest(prices: pd.DataFrame, positions: pd.DataFrame, cost_per_trade: float = 0.0005):
    daily_ret   = prices.pct_change().fillna(0)
    shifted_pos = positions.shift(1).fillna(0)
    n_open      = shifted_pos.sum(axis=1).replace(0, 1)
    strat_ret   = (shifted_pos * daily_ret).sum(axis=1) / n_open
    trades      = shifted_pos.diff().abs().sum(axis=1)
    strat_ret   = strat_ret - trades * cost_per_trade
    strat_cum   = (1 + strat_ret).cumprod() - 1
    return strat_ret, strat_cum

# ─────────────────────────────────────────────────────────────────────
# Main Pipeline (combined chart)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Fetch data
    df = fetch_price_series(TICKERS, START, END, INTERVAL)
    print(f" Price DataFrame shape: {df.shape}")

    # 2) Compute signals
    zs = compute_basket_zscores(df)
    print(f" Z-score DataFrame shape: {zs.shape}")
    pos = generate_positions(zs)
    strat_ret, strat_cum = backtest(df, pos)

    # 3) Benchmark returns
    bench_ret = df.mean(axis=1).pct_change().fillna(0)
    bench_cum = (1 + bench_ret).cumprod() - 1

    # 4) Combined Figure: rebased prices, z-scores, performance
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # 4a) Prices rebased
    rebased = df.div(df.iloc[0]).mul(100)
    rebased.plot(ax=axes[0], title="Prices Rebased to $100")
    axes[0].set_ylabel("Rebased Price")
    axes[0].grid(True)

    # 4b) Z-scores
    zs.plot(ax=axes[1], title="Z-scores vs Basket Mean")
    axes[1].axhline( 1, linestyle='--', color='gray')
    axes[1].axhline(-1, linestyle='--', color='gray')
    axes[1].set_ylabel("Z-score")
    axes[1].grid(True)

    # 4c) Cumulative Performance
    perf = pd.DataFrame({"Strategy": strat_cum, "Basket": bench_cum})
    perf.plot(ax=axes[2], title="Cumulative Performance: Strategy vs Basket")
    axes[2].set_ylabel("Cumulative Return")
    axes[2].grid(True)

    print("Calculating performance metrics…")

    # 5) Print stats
    def ann(c): return (1 + c.iloc[-1]) ** (252 / len(c)) - 1
    def vol(r): return r.std() * (252 ** 0.5)
    s_cagr = ann(strat_cum); b_cagr = ann(bench_cum)
    s_vol  = vol(strat_ret); b_vol  = vol(bench_ret)
    s_sharpe = s_cagr / s_vol if s_vol else float('nan')

    print("\n📊 Performance Summary")
    print(f"Strategy CAGR: {s_cagr:.2%}")
    print(f"Basket    CAGR: {b_cagr:.2%}")
    print(f"Strategy Volatility: {s_vol:.2%}")
    print(f"Basket    Volatility: {b_vol:.2%}")
    print(f"Strategy Sharpe Ratio: {s_sharpe:.2f}")

    plt.tight_layout()
    plt.show()

    
