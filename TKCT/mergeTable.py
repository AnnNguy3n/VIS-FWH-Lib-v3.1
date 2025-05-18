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


def merge_table(db_path: str):
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

        sub_tables = [name for name in table_names if name.startswith(f"T{table_num}_")]
        for sub_table in sub_tables:
            num_opr = int(sub_table.split("_")[1])
            cur_origin.execute(f"select * from {sub_table};")

            bias = sum(NUM_FML_PROCESS[:num_opr])
            inserted_rows = 0

            while True:
                rows = cur_origin.fetchmany(10_000_000)
                if not rows:
                    break

                data_to_insert = [
                    tuple([row[0] + bias, decode(row[1:1+num_opr])] + list(row[1+num_opr:])) for row in rows
                ]
                tmp_string = ",".join(["?"]*len(data_to_insert[0]))
                cur_new.executemany(
                    f"insert into T{table_num} values ({tmp_string});", data_to_insert
                )
                conn_new.commit()

                inserted_rows += len(rows)
                print(f"{new_db_path}: Table T{table_num}, {num_opr} operators, {inserted_rows} rows inserted")

                rows = None
                data_to_insert = None
                gc.collect()

    conn_origin.close()
    conn_new.close()