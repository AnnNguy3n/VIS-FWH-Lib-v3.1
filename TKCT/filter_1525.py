import sqlite3
import numpy as np
from TKCT.mergeTableDB import get_table_names


def fetch_ordered_table_indices(cursor) -> list:
    cursor.execute(get_table_names())
    tables = cursor.fetchall()
    indices = [
        int(name.split("_")[0][1:])
        for (name,) in tables
        if name.startswith("T")
    ]
    indices.sort()
    return indices


def create_intersection_tables(cursor, table_indices: list, column_name: str, threshold: float):
    for idx, table_num in enumerate(table_indices[1:], start=1):
        previous_table = f"T{table_indices[idx-1]}" if idx == 1 else f"TT{table_indices[idx-1]}"
        current_table = f"T{table_num}"
        result_table = f"TT{table_num}"

        cursor.execute(f"""
            CREATE TABLE {result_table} AS
            SELECT cur.*
            FROM (
                SELECT * FROM {current_table} WHERE {column_name} >= ?
            ) AS cur
            INNER JOIN (
                SELECT * FROM {previous_table} WHERE {column_name} >= ?
            ) AS prev
            ON cur.id = prev.id;
        """, (threshold, threshold))


def filter_1525(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table_indices = fetch_ordered_table_indices(cursor)
    assert np.array_equal(
        table_indices,
        np.arange(min(table_indices), max(table_indices) + 1)
    )

    print(f"Processing database: {db_path}")
    print(f"Tables found: {table_indices}")

    create_intersection_tables(cursor, table_indices)
    conn.commit()

    conn.close()
