import pandas as pd, numpy as np
from PySources.base import Base, calculate_formula, convert_strF_to_arrF
import numba as nb
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


THRESHOLD = 0.7
INTEREST = 1.01
VALUEARG_THRESHOLD = 5e8
DATA_PATH = r"Datas\0_to_102_Field_30.xlsx"


def get_grps(ct):
    list_grp = []
    temp = ""
    for c in ct:
        if c in "+-" and temp:
            list_grp.append(temp)
            temp = c
        else:
            temp += c
    list_grp.append(temp)
    return list_grp


@nb.njit
def spearman_coef(A, B):
    n = A.shape[0]
    return 1.0 - 6*sum((A-B)**2) / n / (n**2 - 1)


def loc(ct, vis: Base):
    list_grps = get_grps(ct)
    temp_1 = np.zeros(vis.OPERAND.shape[1], np.float32)
    list_arrs = [
        calculate_formula(convert_strF_to_arrF(grp), vis.OPERAND, temp_1) for grp in list_grps
    ]

    while len(list_grps) != 1:
        list_score = []
        arr_origin = np.sum(list_arrs, axis=0)

        for i in range(len(list_grps)):
            arr_new = np.sum([list_arrs[k] for k in range(len(list_arrs)) if k != i], axis=0)
            arr_origin_rank = arr_origin.argsort().argsort()
            arr_new_rank = arr_new.argsort().argsort()
            corr = spearman_coef(arr_origin_rank, arr_new_rank)
            list_score.append(corr)

        if max(list_score) <= THRESHOLD:
            return "".join(list_grps)
        where = np.argmax(list_score)
        list_grps.pop(where)
        list_arrs.pop(where)

    return list_grps[0]


def run(args):
    name, harngnK, nam_id = args
    data = pd.read_excel(DATA_PATH)
    data = data[data["TIME"] <= nam_id].reset_index(drop=True)
    data.loc[data["PROFIT"] < 0, "PROFIT"] = 0
    vis = Base(data, INTEREST, VALUEARG_THRESHOLD)
    df = pd.read_csv(f"TK_OLD/{name}/HarNgn{harngnK}/{nam_id}.csv")

    list_ct = []
    for ct in df["CT"]:
        list_ct.append(loc(ct, vis))

    list_ct = list(np.unique(list_ct))
    print(name, harngnK, nam_id, len(list_ct))

    path = f"TK_OLD_loc/{name}/HarNgn{harngnK}/{nam_id}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({
        "CT": list_ct
    }).to_csv(path, index=False)


if __name__ == "__main__":
    list_args = []
    for name in ["Classic", "Classic_Stable"]:
        for harngnK in range(2, 9):
            if "Stable" not in name:
                start = 60
            else:
                start = 61

            for nam_id in range(start, 61):
                list_args.append((
                    name, harngnK, nam_id
                ))

    with Pool(processes=cpu_count()-2) as p:
        list(tqdm(p.imap(run, list_args), total=len(list_args)))
