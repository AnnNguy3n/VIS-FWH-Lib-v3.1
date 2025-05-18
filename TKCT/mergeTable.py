import sqlite3
import os
import numpy as np
import gc

# Constants
NUM_FML_PROCESS = [0, 70, 3710, 114380, 2443280, 28126028]


def decode(lst):
    return "_".join(map(str, lst))


def get_table_names():
    return 'SELECT name FROM sqlite_master WHERE type = "table";'


def remove_file(path):
    if os.path.exists(path):
        os.remove(path)


def merge_table(db_path: str, critical_col: str):
    assert db_path.endswith(".db")
    new_db_path = db_path[:-3] + "_new.db"
    remove_file(new_db_path)

    conn_origin = sqlite3.connect(db_path)
    conn_new = sqlite3.connect(new_db_path)
    cur_origin = conn_origin.cursor()
    cur_new = conn_new.cursor()

    # Fetch all table names
    cur_origin.execute(get_table_names())
    table_names = [t[0] for t in cur_origin.fetchall() if t[0].startswith("T")]

    # Extract unique table numbers
    table_nums = np.unique([int(name.split("_")[0][1:]) for name in table_names])

    # Process each main table
    for table_num in table_nums:
        # Get column info from source table T{table_num}_0
        source_table = f"T{table_num}_1"
        cur_origin.execute(f"PRAGMA table_info({source_table})")
        columns_info = cur_origin.fetchall()

        # Construct new table definition, replacing E0 with Formula TEXT
        new_columns = []
        for col in columns_info:
            col_name = col[1]
            col_type = col[2]
            if col_name == "E0":
                new_columns.append("Formula TEXT")
            else:
                new_columns.append(f"{col_name} {col_type}")
        new_table_sql = f"CREATE TABLE T{table_num} ({', '.join(new_columns)});"

        # Create the new table
        cur_new.execute(new_table_sql)
        conn_new.commit()

    conn_origin.close()
    conn_new.close()