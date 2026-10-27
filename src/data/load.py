"""Step 1: convert the raw Instacart CSVs to Parquet.

Parquet is much faster and smaller than CSV (order_products__prior has ~32M rows).

Usage:
    python -m src.data.load
"""
import duckdb

from src.config import PROCESSED_DIR, RAW_DIR, RAW_FILES


def csv_to_parquet() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in RAW_FILES.values() if not (RAW_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {RAW_DIR}: {missing}. "
            "Download the Instacart dataset from Kaggle (see README)."
        )

    con = duckdb.connect()
    for name, filename in RAW_FILES.items():
        src = (RAW_DIR / filename).as_posix()
        dst = (PROCESSED_DIR / f"{name}.parquet").as_posix()
        con.execute(
            f"COPY (SELECT * FROM read_csv_auto('{src}', header=true)) "
            f"TO '{dst}' (FORMAT PARQUET)"
        )
        rows = con.execute(f"SELECT COUNT(*) FROM '{dst}'").fetchone()[0]
        print(f"{name:<22} {rows:>12,} rows -> {dst}")
    con.close()


if __name__ == "__main__":
    csv_to_parquet()
