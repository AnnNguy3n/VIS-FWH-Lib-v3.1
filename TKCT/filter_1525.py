import sqlite3
import numpy as np


def create_intersection_tables(cursor, table_indices: list, column_name: str, threshold: float):
    for idx, table_num in enumerate(table_indices[1:], start=1):
        previous_table = f"origin.T{table_indices[idx-1]}" if idx == 1 else f"TT{table_indices[idx-1]}"
        current_table = f"origin.T{table_num}"
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


def filter_1525(db_path: str, critical_col: str, threshold: float):
    assert db_path.endswith(".db")
    new_db_path = db_path[:-3] + f"_{critical_col}.db"

    conn_new = sqlite3.connect(new_db_path)
    cursor_new = conn_new.cursor()

    # Attach database gốc với alias 'origin'
    cursor_new.execute(f"ATTACH DATABASE '{db_path}' AS origin")

    # Lấy danh sách bảng từ origin
    cursor_new.execute("SELECT name FROM origin.sqlite_master WHERE type='table'")
    tables = [name[0] for name in cursor_new.fetchall() if name[0].startswith("T")]
    table_indices = sorted(set(int(name.split("_")[0][1:]) for name in tables))

    create_intersection_tables(cursor_new, table_indices, critical_col, threshold)

    conn_new.commit()
    conn_new.close()
