"""Build the DuckDB marts from data/raw (sql/marts.sql) and export them to data/marts as parquet."""
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
MARTS = ["mart_merchant_week", "mart_kpi_week", "mart_merchant_h1", "mart_reorder_customer", "mart_first_order"]


def main():
    os.chdir(ROOT)
    (ROOT / "data" / "marts").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute((ROOT / "sql" / "marts.sql").read_text(encoding="utf-8"))
    for t in MARTS:
        con.execute(f"COPY {t} TO 'data/marts/{t}.parquet' (FORMAT PARQUET)")
        print(f"{t:24s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>9,} rows")


if __name__ == "__main__":
    main()
