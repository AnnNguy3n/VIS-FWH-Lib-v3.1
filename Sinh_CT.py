import numpy as np
import numba as nb


INTEREST = 1.01
VALUEARG_THRESHOLD = 5e8


@nb.njit
def M_invest_method(
    weight, profit, symbol, sufficient_liquidity,
    threshold, threshold_cycle_idx,
    NUM_SYMBOL_UNIQUE, NUM_STRATEGY,
    INDEX, INDEX_LEN, INTEREST, NUM_CYCLE_RESULT
):
    symbol_streak = np.zeros(NUM_SYMBOL_UNIQUE, np.int32)
    tmp_profit  = np.zeros(NUM_STRATEGY, np.float32)
    tmp_harmean = np.zeros(NUM_STRATEGY, np.float32)
    tmp_icount  = np.zeros(NUM_STRATEGY, np.int32)

    result = np.zeros(NUM_STRATEGY, np.float32)

    # Khởi tạo
    market_streak = 0

    # Duyệt ngược qua từng chu kỳ
    for cycle_idx in range(INDEX_LEN - 2, 0, -1):
        start = INDEX[cycle_idx]
        end   = INDEX[cycle_idx + 1]

        # Reset bộ đếm đầu tư cho chu kỳ hiện tại
        N = INDEX_LEN - 1 - cycle_idx
        num_applicable_strategy = min(N, NUM_STRATEGY)
        any_pass_threshold = False

        tmp_profit[:] = 0
        tmp_icount[:] = 0

        # Duyệt qua tất cả công ty trong chu kỳ hiện tại
        for i in range(start, end):
            sym = symbol[i]
            if weight[i] <= threshold:
                symbol_streak[sym] = 0
                continue

            any_pass_threshold = True
            symbol_streak[sym] += 1
            sym_streak = symbol_streak[sym]

            if not sufficient_liquidity[i]: continue

            # Update cho từng strategy k (tương ứng độ dài streak yêu cầu)
            p = profit[i]
            for k in range(num_applicable_strategy):
                if sym_streak > min(market_streak, k):
                    tmp_profit[k] += p
                    tmp_icount[k] += 1

        # Cập nhật harmean tích luỹ
        for k in range(num_applicable_strategy):
            denom = (tmp_profit[k] / tmp_icount[k]) if tmp_icount[k] != 0 else INTEREST
            tmp_harmean[k] += 1.0 / (denom + 1e-9)

        # Lưu kết quả nếu cycle nằm trong vùng cần ghi
        if cycle_idx <= NUM_CYCLE_RESULT and threshold_cycle_idx + 1 >= cycle_idx:
            offset = (NUM_CYCLE_RESULT - cycle_idx) * NUM_STRATEGY
            for k in range(num_applicable_strategy):
                result[offset + k] = (N - k) / tmp_harmean[k]

        # Cập nhật market streak
        market_streak = (market_streak + 1) if any_pass_threshold else 0

    return result


from PySources.base import Base, calculate_formula, calculate_formula_v2, convert_strF_to_arrF, NEG_INF_FLOAT32, encode_formula


def get_info_ct(ct_: str, vis: Base, eval_method: int):
    ct = convert_strF_to_arrF(ct_)
    if eval_method == 0: calculator = calculate_formula
    elif eval_method == 1: calculator = calculate_formula_v2

    temp_1 = np.zeros(vis.OPERAND.shape[1], np.float32)
    weight = calculator(ct, vis.OPERAND, temp_1)

    thresholds = np.full(
        shape=5 * (vis.INDEX.shape[0] - 2),
        fill_value=NEG_INF_FLOAT32
    )

    for i in range(vis.INDEX.shape[0] - 2):
        start = vis.INDEX[i + 1]
        end = vis.INDEX[i + 2]
        arr = np.unique(weight[start:end])
        arr[::-1].sort()
        thresholds[5*i : 5*i + min(5, len(arr))] = arr[: min(5, len(arr))]

    num_symbol_unique = np.unique(vis.SYMBOL).size

    list_result = []
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        result = M_invest_method(
            weight, vis.PROFIT, vis.SYMBOL, vis.BOOL_ARG,
            threshold, i // 5,
            num_symbol_unique, 9,
            vis.INDEX, vis.INDEX.size,
            INTEREST, 1
        )
        list_result.append(result)

    arr = np.vstack(list_result)
    max_idx = arr.argmax(axis=0)
    best_scores = arr[max_idx, np.arange(arr.shape[1])]
    best_thresholds = thresholds[max_idx]

    info = [0, "_".join(map(str, encode_formula(ct, vis.OPERAND.shape[0])))]
    for s in range(arr.shape[1]):
        info.extend([best_thresholds[s], best_scores[s]])

    return info


import re
import os
import pandas as pd


def process(path):
    name = re.search(r"\d+.csv", path).group()
    cycle_num = int(name.replace(".csv", ""))

    if "Root" in path:
        eval_method = 1
    elif "Classic" in path:
        eval_method = 0

    data = pd.read_excel(r"Datas\0_to_102_Field_30.xlsx")
    data = data[data["TIME"] <= cycle_num].reset_index(drop=True)
    data.loc[data["PROFIT"] < 0, "PROFIT"] = 0
    vis = Base(data, INTEREST, VALUEARG_THRESHOLD)
    df_CT = pd.read_csv(path)

    list_info = []
    for ct in df_CT["CT"]:
        list_info.append(get_info_ct(ct, vis, eval_method))

    list_col = ["id", "CT"]
    for i in range(9):
        list_col.extend([f"ValNgn{i}", f"HarNgn{i}"])
    df = pd.DataFrame(list_info, columns=list_col)
    harNgnK = re.search(r"HarNgn\d", path).group()
    df.sort_values(harNgnK, ascending=False, inplace=True, ignore_index=True)
    df.to_excel(path.replace(name, f"{cycle_num-60}.xlsx"), index=False)


list_args = []
for dirpath, dirnames, filenames in os.walk("TK_OLD_loc"):
    for filename in filenames:
        if filename.endswith(".csv"):
            path = os.path.join(dirpath, filename)
            list_args.append(path)


from multiprocessing import Pool, cpu_count
from tqdm import tqdm


if __name__ == "__main__":
    with Pool(processes=cpu_count()-2) as p:
        list(tqdm(p.imap(process, list_args), total=len(list_args)))
