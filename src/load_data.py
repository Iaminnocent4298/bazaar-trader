import glob

import pandas as pd


def load_all():
    """Load every daily CSV in data/ into one table, sorted by time."""
    files = sorted(glob.glob("data/bazaar_2*.csv"))
    if not files:
        raise FileNotFoundError("No data files found in data/")
    frames = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["product_id", "timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    df = load_all()
    print("Rows:", len(df))
    print("Items:", df["product_id"].nunique())
    print("From:", df["timestamp"].min())
    print("To:  ", df["timestamp"].max())