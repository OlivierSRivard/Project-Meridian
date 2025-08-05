import pandas as pd

def compute_basket_zscores(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score of each stock vs the basket mean on each date.
    Gaps are forward-filled so the output is never empty.
    """
    # forward-fill small gaps, then drop any row that is still all-NaN
    clean = price_df.ffill().dropna(how="all")

    basket_mean = clean.mean(axis=1)
    basket_std  = clean.std(axis=1)

    zscores = clean.sub(basket_mean, axis=0).div(basket_std, axis=0)
    return zscores
