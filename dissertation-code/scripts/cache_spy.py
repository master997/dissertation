from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config_loader import load_config


def main() -> None:
    config = load_config(Path.cwd() / "config.yaml")
    cache_path = Path(config.data_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        config.ticker,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("yfinance returned no data; check network and ticker/date range.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    print(f"Wrote {len(df)} rows to {cache_path}")
    print(df.tail())


if __name__ == "__main__":
    main()

