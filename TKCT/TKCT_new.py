import numpy as np
import numba as nb
import sqlite3
from tqdm import tqdm
from PySources.base import Base, calculate_formula, decode_formula, convert_arrF_to_strF, calculate_formula_v2


__NUM_THRESHOLD_PER_CYCLE__ = 10


@nb.njit
def SingleYear_Har_Test(
    INDEX: np.ndarray,
    PROFIT: np.ndarray,
    INTEREST: float,
    bool_invest: np.ndarray
):
    size = INDEX.size - 1
    temp_profit_1 = 0.0
    for i in range(size-1, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        valid_mask = bool_invest[start:end]
        if not np.any(valid_mask):
            temp_profit_1 += 1.0 / INTEREST
        else:
            temp_profit_1 += 1.0 / np.mean(PROFIT[start:end][valid_mask])
    return (size - 1) / temp_profit_1


@nb.njit
def DoubleYear_Har_Test(
    INDEX: np.ndarray,
    PROFIT: np.ndarray,
    SYMBOL: np.ndarray,
    INTEREST: float,
    bool_wgt: np.ndarray,
    bool_invest: np.ndarray
):
    size = INDEX.size - 1
    temp_profit_2 = 0.0
    start, end = INDEX[size-1], INDEX[size]
    if not np.any(bool_wgt[start:end]):
        reason = 1
    else:
        reason = 0
    for i in range(size-2, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        cur_cyc_mask = bool_invest[start:end]
        if reason == 0:
            cur_cyc_sym = SYMBOL[start:end]
            end2 = INDEX[i+2]
            pre_cyc_mask = bool_wgt[end:end2]
            pre_cyc_sym = SYMBOL[end:end2]
            coms = np.intersect1d(cur_cyc_sym[cur_cyc_mask], pre_cyc_sym[pre_cyc_mask])
            isin = np.isin(cur_cyc_sym, coms)
            lst_pro = PROFIT[start:end][isin]
        else:
            lst_pro = PROFIT[start:end][cur_cyc_mask]

        if len(lst_pro) == 0:
            temp_profit_2 += 1.0 / INTEREST
        else:
            temp_profit_2 += 1.0 / np.mean(lst_pro)

        if not np.any(bool_wgt[start:end]):
            reason = 1
        else:
            reason = 0

    return (size - 2) / temp_profit_2


@nb.njit
def TripleYear_Har_Test(
    INDEX: np.ndarray,
    PROFIT: np.ndarray,
    SYMBOL: np.ndarray,
    INTEREST: float,
    bool_wgt: np.ndarray,
    bool_invest: np.ndarray
):
    size = INDEX.size - 1
    temp_profit_3 = 0.0
    start, end, end2 = INDEX[size-2], INDEX[size-1], INDEX[size]
    if not np.any(bool_wgt[start:end]):
        reason = 2
    else:
        if not np.any(bool_wgt[end:end2]):
            reason = 1
        else:
            reason = 0
    for i in range(size-3, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        cur_cyc_mask = bool_invest[start:end]
        if reason == 2:
            lst_pro = PROFIT[start:end][cur_cyc_mask]
        else:
            end2 = INDEX[i+2]
            pre_cyc_mask = bool_wgt[end:end2]
            cur_cyc_sym = SYMBOL[start:end]
            pre_cyc_sym = SYMBOL[end:end2]
            coms = np.intersect1d(cur_cyc_sym[cur_cyc_mask], pre_cyc_sym[pre_cyc_mask])
            if reason == 0:
                end3 = INDEX[i+3]
                pre2_cyc_mask = bool_wgt[end2:end3]
                pre2_cyc_sym = SYMBOL[end2:end3]
                coms = np.intersect1d(coms, pre2_cyc_sym[pre2_cyc_mask])

            isin = np.isin(cur_cyc_sym, coms)
            lst_pro = PROFIT[start:end][isin]

        if len(lst_pro) == 0:
            temp_profit_3 += 1.0 / INTEREST
        else:
            temp_profit_3 += 1.0 / np.mean(lst_pro)

        if not np.any(bool_wgt[start:end]):
            reason = 2
        else:
            if reason == 2: reason = 1
            else: reason = 0

    return (size - 3) / temp_profit_3


@nb.njit
def MixedSingleDoubleTriple(
    WEIGHT: np.ndarray,
    INDEX: np.ndarray,
    PROFIT: np.ndarray,
    SYMBOL: np.ndarray,
    INTEREST: float,
    BOOL_ARG: np.ndarray
):
    size = INDEX.size - 1
    arr_loop = np.full((size-1)*__NUM_THRESHOLD_PER_CYCLE__, -1.7976931348623157e+308, float)
    for i in range(1, size):
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = np.unique(WEIGHT[start:end])
        wgt_[::-1].sort()
        start = __NUM_THRESHOLD_PER_CYCLE__*(i-1)
        if len(wgt_) < __NUM_THRESHOLD_PER_CYCLE__:
            arr_loop[start:start+len(wgt_)] = wgt_
        else:
            arr_loop[start:start+__NUM_THRESHOLD_PER_CYCLE__] = wgt_[:__NUM_THRESHOLD_PER_CYCLE__]

    ValHar1 = -1.0
    HarNgn1 = -1.0

    ValHar2 = -1.0
    HarNgn2 = -1.0

    ValHar3 = -1.0
    HarNgn3 = -1.0

    arr_check = np.full((size-1)*__NUM_THRESHOLD_PER_CYCLE__, True)
    for ii in range(len(arr_loop)):
        if not arr_check[ii]: continue
        v = arr_loop[ii]
        arr_check[arr_loop == v] = False

        bool_wgt = WEIGHT > v
        bool_invest = bool_wgt & BOOL_ARG
        Har1 = SingleYear_Har_Test(INDEX, PROFIT, INTEREST, bool_invest)
        Har2 = DoubleYear_Har_Test(INDEX, PROFIT, SYMBOL, INTEREST, bool_wgt, bool_invest)
        Har3 = TripleYear_Har_Test(INDEX, PROFIT, SYMBOL, INTEREST, bool_wgt, bool_invest)

        if Har1 > HarNgn1:
            HarNgn1 = Har1
            ValHar1 = v

        if Har2 > HarNgn2:
            HarNgn2 = Har2
            ValHar2 = v

        if Har3 > HarNgn3:
            HarNgn3 = Har3
            ValHar3 = v

    return ValHar1, HarNgn1, ValHar2, HarNgn2, ValHar3, HarNgn3


@nb.njit
def SingleYear_Har_Invest(
    WEIGHT: np.ndarray,
    BOOL_ARG: np.ndarray,
    threshold: float
):
    bool_wgt = WEIGHT > threshold
    return bool_wgt & BOOL_ARG


@nb.njit
def DoubleYear_Har_Invest(
    WEIGHT: np.ndarray,
    INDEX: np.ndarray,
    SYMBOL: np.ndarray,
    BOOL_ARG: np.ndarray,
    threshold: float
):
    size = INDEX.size - 1
    bool_wgt = WEIGHT > threshold
    bool_invest = bool_wgt & BOOL_ARG
    list_invest = np.full(WEIGHT.shape[0], False, dtype=bool)
    start, end = INDEX[size-1], INDEX[size]
    if not np.any(bool_wgt[start:end]):
        reason = 1
    else:
        reason = 0
    for i in range(size-2, -1, -1):
        start, end = INDEX[i], INDEX[i+1]
        cur_cyc_mask = bool_invest[start:end]
        if reason == 0:
            cur_cyc_sym = SYMBOL[start:end]
            end2 = INDEX[i+2]
            pre_cyc_mask = bool_wgt[end:end2]
            pre_cyc_sym = SYMBOL[end:end2]
            coms = np.intersect1d(cur_cyc_sym[cur_cyc_mask], pre_cyc_sym[pre_cyc_mask])
            isin = np.isin(cur_cyc_sym, coms)
            list_invest[start:end] = isin
        else:
            list_invest[start:end] = cur_cyc_mask

        if not np.any(bool_wgt[start:end]):
            reason = 1
        else:
            reason = 0

    return list_invest


@nb.njit
def TripleYear_Har_Invest(
    WEIGHT: np.ndarray,
    INDEX: np.ndarray,
    SYMBOL: np.ndarray,
    BOOL_ARG: np.ndarray,
    threshold: float
):
    size = INDEX.size - 1
    bool_wgt = WEIGHT > threshold
    bool_invest = bool_wgt & BOOL_ARG
    list_invest = np.full(WEIGHT.shape[0], False, dtype=bool)
    start, end, end2 = INDEX[size-2], INDEX[size-1], INDEX[size]
    if not np.any(bool_wgt[start:end]):
        reason = 2
    else:
        if not np.any(bool_wgt[end:end2]):
            reason = 1
        else:
            reason = 0
    for i in range(size-3, -1, -1):
        start, end = INDEX[i], INDEX[i+1]
        cur_cyc_mask = bool_invest[start:end]
        if reason == 2:
            list_invest[start:end] = cur_cyc_mask
        else:
            end2 = INDEX[i+2]
            pre_cyc_mask = bool_wgt[end:end2]
            cur_cyc_sym = SYMBOL[start:end]
            pre_cyc_sym = SYMBOL[end:end2]
            coms = np.intersect1d(cur_cyc_sym[cur_cyc_mask], pre_cyc_sym[pre_cyc_mask])
            if reason == 0:
                end3 = INDEX[i+3]
                pre2_cyc_mask = bool_wgt[end2:end3]
                pre2_cyc_sym = SYMBOL[end2:end3]
                coms = np.intersect1d(coms, pre2_cyc_sym[pre2_cyc_mask])

            isin = np.isin(cur_cyc_sym, coms)
            list_invest[start:end] = isin

        if not np.any(bool_wgt[start:end]):
            reason = 2
        else:
            if reason == 2: reason = 1
            else: reason = 0

    return list_invest


def get_info_invest(vis: Base, ct_, eval_method: str, temp_1: np.ndarray, exclude_threshold=2e-6):
    """
    Parameters
    ----------
    vis : class Base
    eval_method : {'classic', 'root'} optional
    temp_1 : array float64 (size = vis.OPERAND.shape[1])
    exclude_threshold : float (func will return None if max(weight) - min(weight) <= exclude_threshold)
    """
    ct = list(map(int, ct_.split("_")))
    ct = decode_formula(np.array(ct), vis.OPERAND.shape[0])
    if eval_method == "classic":
        weight = calculate_formula(ct, vis.OPERAND, temp_1)
    elif eval_method == "root":
        weight = calculate_formula_v2(ct, vis.OPERAND, temp_1)
    else: raise
    if abs(weight.max() - weight.min()) <= exclude_threshold:
        return None
    Val1, Har1, Val2, Har2, Val3, Har3 = MixedSingleDoubleTriple(
        weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG
    )
    list_invest_1 = SingleYear_Har_Invest(weight, vis.BOOL_ARG, Val1)
    list_invest_2 = DoubleYear_Har_Invest(weight, vis.INDEX, vis.SYMBOL, vis.BOOL_ARG, Val2)
    list_invest_3 = TripleYear_Har_Invest(weight, vis.INDEX, vis.SYMBOL, vis.BOOL_ARG, Val3)

    size = vis.INDEX.shape[0] - 1
    list_result_1 = []
    list_result_2 = []
    list_result_3 = []

    for i in range(size-1, 0, -1):
        start, end = vis.INDEX[i], vis.INDEX[i+1]
        invest = np.where(list_invest_1[start:end])[0]
        w_ = weight[invest]
        arg = np.argsort(w_, kind="stable")[::-1]
        invest = invest[arg]
        list_result_1.append("".join(map(chr, invest)))

        if i <= size - 2:
            invest = np.where(list_invest_2[start:end])[0]
            w_ = weight[invest]
            arg = np.argsort(w_, kind="stable")[::-1]
            invest = invest[arg]
            list_result_2.append("".join(map(chr, invest)))

        if i <= size - 3:
            invest = np.where(list_invest_3[start:end])[0]
            w_ = weight[invest]
            arg = np.argsort(w_, kind="stable")[::-1]
            invest = invest[arg]
            list_result_3.append("".join(map(chr, invest)))

    rs = {
        0: ct,
        1: np.array(list_result_1 + list_result_2 + list_result_3)
    }
    rs[2] = rs[1] == ""
    return rs


def compare(A, B):
    a = np.count_nonzero(A[1] == B[1])
    b = np.count_nonzero(A[2] & B[2])
    return a - b


def filter(vis: Base,
           DB_PATH, NAM_ID, target, rate, FOLDER_SAVE, critical_col,
           eval_method: str, exclude_threshold=2e-6, add_after_filename=""):
    connect = sqlite3.connect(DB_PATH)
    cursor = connect.cursor()

    cursor.execute(f"SELECT count(*) FROM T{NAM_ID};")
    num_rows = cursor.fetchall()[0][0]
    temp_1 = np.empty(vis.OPERAND.shape[1], np.float64)
    temp_info = get_info_invest(vis, "+0", eval_method, temp_1, 0.0)
    total = len(temp_info[1])

    threshold = total * rate
    list_save = []
    cursor.execute(f"SELECT * FROM T{NAM_ID} ORDER BY {critical_col} ASC;")
    # with tqdm(total=num_rows) as pbar:
    if True:
        for i in range(num_rows):
            temp = cursor.fetchone()
            ct = temp[1]
            info = get_info_invest(vis, ct, eval_method, temp_1, exclude_threshold)
            if info is None:
                # pbar.update(1)
                continue
            check = True
            for k in range(len(list_save)):
                if compare(list_save[k], info) >= threshold:
                    check = False
                    break
            if check:
                list_save.append(info)
            #     pbar.set_postfix(saved = len(list_save))

            # pbar.update(1)

    if len(list_save) > target:
        list_save_final = list_save[-target:]
    else:
        list_save_final = list_save

    with open(f"{FOLDER_SAVE}/{NAM_ID}{add_after_filename}.txt", "w") as f:
        f.write("\n".join(map(lambda x: convert_arrF_to_strF(x[0]), list_save_final[::-1])))


### END FILE ###