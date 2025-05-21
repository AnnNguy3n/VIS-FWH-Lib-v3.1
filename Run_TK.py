from TKCT.TK_detail import get_dfs
from PySources.base import Base
import pandas as pd
import os
import multiprocessing as mp
from tqdm import tqdm


MIN_CYC = 60
NUM_CYC = 42
INTEREST = 1.01
VALUEARG_THRESHOLD = 5e8
DATA_PATH = "/Users/annnguy3n/Desktop/VIS-FWH-Lib-v3/Datas/0_to_101_Field_30.xlsx"
FOLDER_CT = "/Users/annnguy3n/Desktop/VIS-FWH-Lib-v3/FMLs"


def run(args: list[str]):
    folder_save, cyc_id = args
    if folder_save.__contains__("Classic/HarNgn"):
        eval_method = "classic"
    elif folder_save.__contains__("Root/HarNgn"):
        eval_method = "root"
    else: raise

    center_method_num = int(folder_save[-1])

    df_CT = pd.read_excel(f"{folder_save}/{cyc_id}.xlsx")

    data = pd.read_excel(DATA_PATH)
    data = data[data["TIME"] <= MIN_CYC + cyc_id]
    vis = Base(data, INTEREST, VALUEARG_THRESHOLD)

    df_info, df_sum_rank = get_dfs(vis, df_CT, eval_method, center_method_num)
    df_info.to_csv(f"{folder_save}/{MIN_CYC + cyc_id}.csv", index=False)
    df_sum_rank.to_csv(f"{folder_save}/SUM_RANK{MIN_CYC + cyc_id}.csv", index=False)


if __name__ == "__main__":
    list_args = []
    for eval_method in ["Classic", "Root"]:
        for center_method_num in range(2, 9):
            for cyc_id in range(42):
                list_args.append((
                    f"{FOLDER_CT}/{eval_method}/HarNgn{center_method_num}", cyc_id
                ))

    with mp.Pool(processes=mp.cpu_count()) as p:
        list(tqdm(p.imap(run, list_args), total=len(list_args)))
