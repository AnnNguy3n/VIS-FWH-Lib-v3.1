import sqlite3
import os
import pandas as pd
import numpy as np
from TKCT.mergeTable import get_table_names


def sample_data_with_rate(data: pd.DataFrame, rate: float) -> pd.DataFrame:
    n = int(np.ceil(len(data) * rate))
    return data.sample(n=n)


def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)


def fetch_table_names(cursor, filter_1525: bool) -> list:
    cursor.execute(get_table_names())
    tables = cursor.fetchall()
    if filter_1525:
        return [int(tb[0][2:]) for tb in tables if tb[0].startswith("TT")]
    return [int(tb[0][1:]) for tb in tables if tb[0].startswith("T") and not tb[0].startswith("TT")]


def build_temp_db_path(db_path: str, filter_1525: bool) -> str:
    suffix = "f_1525_temp.db" if filter_1525 else "f_temp.db"
    return db_path[:-8] + suffix


def filter_unique_profit_value(db_path: str, critical_col: str, target: int = 100_000, filter_1525: bool = False):
    assert db_path.endswith("f_new.db")

    db_temp = build_temp_db_path(db_path, filter_1525)
    remove_file(db_temp)

    conn_temp = sqlite3.connect(db_temp)
    cursor_temp = conn_temp.cursor()
    conn_origin = sqlite3.connect(db_path)
    cursor_origin = conn_origin.cursor()

    table_indices = fetch_table_names(cursor_origin, filter_1525)
    print(f"Working on: {db_temp}, Tables: {table_indices}")

    # Create tables in temporary database
    for idx in table_indices:
        table_name = f"TT{idx}" if filter_1525 else f"T{idx}"

        # Lấy cấu trúc bảng từ database gốc
        cursor_origin.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor_origin.fetchall()

        # Tạo câu lệnh CREATE TABLE
        columns_def = ", ".join([f"{col[1]} {col[2]}" for col in columns_info])
        create_sql = f"CREATE TABLE {table_name} ({columns_def});"

        # Tạo bảng trong database tạm
        cursor_temp.execute(create_sql)
    conn_temp.commit()

    list_col_name = [col[1] for col in columns_info]
    print(list_col_name)