import numpy as np
import numba as nb
import sqlite3
from tqdm import tqdm
from PySources.base import Base, NEG_INF_FLOAT32, decode_formula,\
calculate_formula, calculate_formula_v2, convert_arrF_to_strF
from scipy import stats
from scipy.stats import rankdata
import pandas as pd


# @nb.njit
# def StreakInvest(
#     WEIGHT: np.ndarray,
#     INDEX: np.ndarray,
#     SYMBOL: np.ndarray,
#     BOOL_ARG: np.ndarray,
#     threshold: float,
#     method_num: int,
#     num_symbol_unique: int
# ):
#     size = INDEX.size - 1
#     array_invest = np.full(shape=WEIGHT.shape[0], fill_value=False, dtype=np.bool_)
#     symbol_streak = np.full(shape=num_symbol_unique, fill_value=0, dtype=np.int32)
#     market_streak = 0

#     for k in range(size-2, -1, -1):
#         start, end = INDEX[k], INDEX[k+1]
#         any_pass_threshold = False
#         N = size - 1 - k

#         for i in range(start, end):
#             sym = SYMBOL[i]
#             if WEIGHT[i] <= threshold:
#                 symbol_streak[sym] = 0
#                 continue

#             any_pass_threshold = True
#             symbol_streak[sym] += 1
#             sym_streak = symbol_streak[sym]

#             if not BOOL_ARG[i]: continue

#             if N >= method_num:
#                 if sym_streak > min(market_streak, method_num-1):
#                     array_invest[i] = True

#         if any_pass_threshold: market_streak +=1
#         else: market_streak = 0

#     return array_invest


def LinRegressPreProfit(vis: Base, weight):
    arr = weight[vis.INDEX[1]:].copy()
    profit = vis.PROFIT[vis.INDEX[1]:].copy()
    list_x = []
    list_y = []
    arg_ = np.argsort(arr)[::-1]
    arr = arr[arg_]
    profit = profit[arg_]

    for v in np.unique(arr):
        if v == NEG_INF_FLOAT32: continue
        idx = np.where(arr==v)[0][-1]
        list_x.append(idx+1)
        list_y.append(profit[:idx+1].mean())

    rs = stats.linregress(list_x, list_y)
    return rs.slope, rs.intercept


def get_info(
    vis: Base,
    df_CT: pd.DataFrame,
    ct_idx: int,
    sum_rank: np.ndarray,
    sum_rank_ni: np.ndarray,
    num_field: int,
    # temp_1_arr: np.ndarray,
    method: str,
    arr_streakInvest_method: np.ndarray,
    # num_symbol_unique: int,
    weight: np.ndarray,
    arr_list_invest,
    # eval_method="classic"
):
    ct = df_CT.loc[ct_idx, "CT"]
    ct = decode_formula(np.array(list(map(int, ct.split("_")))), num_field)
    # if eval_method == "classic":
    #     weight = calculate_formula(ct, vis.OPERAND, temp_1_arr)
    # elif eval_method == "root":
    #     weight = calculate_formula_v2(ct, vis.OPERAND, temp_1_arr)
    # else: raise

    #
    # for i in range(vis.INDEX.shape[0]-1):
    #     start, end = vis.INDEX[i], vis.INDEX[i+1]
    #     ValMethod = df_CT.loc[ct_idx, method]
    #     wgt = list(weight[start:end]) + [ValMethod]
    #     ranks = rankdata(np.array(wgt), method="min") - 1
    #     sum_rank[start:end] += ranks[:-1]
    #     sum_rank_ni[i] += ranks[-1]

    # slope, intercept = LinRegressPreProfit(vis, weight)
    slope, intercept = 0, 0

    #
    start, end = vis.INDEX[0], vis.INDEX[1]
    result = {
        "CT": convert_arrF_to_strF(ct),
        "Slope": slope,
        "Intercept": intercept
    }
    min_method = min(arr_streakInvest_method)
    for i in arr_streakInvest_method:
        ValM, HarM = df_CT.loc[ct_idx, [f"ValHar{i}", f"HarNgn{i}"]]
        # list_invest = StreakInvest(
        #     weight, vis.INDEX, vis.SYMBOL, vis.BOOL_ARG, ValM, i, num_symbol_unique
        # )
        list_invest = arr_list_invest[i-min_method]
        invest = np.where(list_invest[start:end])[0]
        w_ = weight[invest]
        arg = np.argsort(w_, kind="stable")[::-1]
        invest = invest[arg] + start
        w_ = w_[arg]
        Cty1 = "_".join(map(lambda x: vis.symbol_name[vis.SYMBOL[x]], invest))
        if Cty1 == "":
            Pro1 = vis.INTEREST
        else:
            Pro1 = np.mean(vis.PROFIT[invest])
        Values_1 = "_".join(map(str, w_[arg]))
        result.update({
            f"ValHar{i}": ValM,
            f"HarNgn{i}": HarM,
            f"CtyNgn{i}": Cty1,
            f"Values{i}": Values_1,
            f"ProNgn{i}": Pro1
        })

    return result


def get_dfs(vis: Base, df_CT: pd.DataFrame, center_method_num: int, N_weights, N_list_invest):
    arr_streakInvest_method = np.arange(center_method_num-1, center_method_num+2)
    # temp_1_arr = np.zeros(vis.OPERAND.shape[1], dtype=np.float32)
    # num_symbol_unique = np.unique(vis.SYMBOL).size
    list_data = []
    sum_rank = np.zeros(vis.data.shape[0])
    sum_rank_ni = np.zeros(vis.INDEX.shape[0]-1)
    method = f"ValHar{center_method_num}"
    for i in range(df_CT.shape[0]):
        # print(i)
        idx_start = vis.OPERAND.shape[1] * i
        idx_end = vis.OPERAND.shape[1] * (i + 1)
        weight = N_weights[idx_start:idx_end]
        arr_list_invest = N_list_invest[:, idx_start:idx_end]
        info = get_info(
            vis, df_CT, i, sum_rank, sum_rank_ni, vis.OPERAND.shape[0],
            method, arr_streakInvest_method, weight, arr_list_invest
        )
        list_data.append(info)

    list_syms = []
    list_value_rank = []
    list_year = []

    for i in range(vis.INDEX.shape[0]-1):
        # print(i)
        start, end = vis.INDEX[i], vis.INDEX[i+1]
        list_syms.append("NOT_INVEST")
        list_syms.extend(vis.data.iloc[start:end]["SYMBOL"].to_list())

        list_value_rank.append(sum_rank_ni[i])
        list_value_rank.extend(list(sum_rank[start:end]))

        list_year.extend([vis.data["TIME"].max()-i]*(end-start+1))

    df_sum_rank = pd.DataFrame({
        "SYMBOL": list_syms,
        "SUM_RANK": list_value_rank,
        "TIME": list_year
    })

    df_info = pd.DataFrame(list_data)

    return df_info, df_sum_rank
