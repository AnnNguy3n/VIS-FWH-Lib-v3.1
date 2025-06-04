import sqlite3
import pandas as pd
import numpy as np


def sample_data_with_rate(data: pd.DataFrame, rate: float) -> pd.DataFrame:
    n = int(np.ceil(len(data) * rate))
    return data.sample(n=n)


def filter_unique_profit_value(
    db_path: str, critical_col: str, nam_id: int,
    target: int = 100_000, filter_1525: bool = False, threshold = 0.0
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table_name = f"TT{nam_id}" if filter_1525 else f"T{nam_id}"
    cursor.execute(f"PRAGMA table_info({table_name})")
    list_col = [r[1] for r in cursor.fetchall()]
    assert critical_col in list_col

    cursor.execute(f"SELECT COUNT(*) FROM {table_name} where {critical_col} >= {threshold};")
    num_rows = cursor.fetchone()[0]
    sample_rate = min(float(target) / num_rows, 1.0)

    # Fetch data in chunks
    cursor.execute(f"SELECT * FROM {table_name} where {critical_col} >= {threshold};")
    list_df = []
    while True:
        batch = cursor.fetchmany(1_000_000)
        if not batch:
            break

        batch_df = pd.DataFrame(batch)
        batch_df.columns = list_col
        batch_df["temp"] = batch_df[critical_col].round(3)

        sampled_batch = batch_df.groupby("temp", group_keys=False).apply(
            lambda x: sample_data_with_rate(x, sample_rate)
        )
        list_df.append(sampled_batch)

    # Merge all sampled batches
    data = pd.concat(list_df, ignore_index=True)

    # Re-sample if needed
    final_sample_rate = min(float(target) / len(data), 1.0)
    final_sample = data.groupby("temp", group_keys=False).apply(
        lambda x: sample_data_with_rate(x, final_sample_rate)
    ).drop(columns=["temp"])

    conn.close()
    return final_sample
