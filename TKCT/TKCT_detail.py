from TKCT.TKCT_new import MixedSingleDoubleTriple, SingleYear_Har_Invest, DoubleYear_Har_Invest, TripleYear_Har_Invest
from PySources.base import Base, convert_strF_to_arrF, calculate_formula, calculate_formula_v2
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from scipy.stats import rankdata


def LinRegressPreProfit(vis: Base, weight):
    arr = weight[vis.INDEX[1]:].copy()
    profit = vis.PROFIT[vis.INDEX[1]:].copy()
    list_x = []
    list_y = []
    arg_ = np.argsort(arr)[::-1]
    arr = arr[arg_]
    profit = profit[arg_]

    for v in np.unique(arr):
        if v == -1.7976931348623157e+308: continue
        idx = np.where(arr==v)[0][-1]
        list_x.append(idx+1)
        list_y.append(profit[:idx+1].mean())

    rs = stats.linregress(list_x, list_y)
    return rs.slope, rs.intercept


def get_info(vis: Base, ct_: str, method: str, sum_rank: np.ndarray, sum_rank_ni: np.ndarray, eval_method="classic", args = []):
    ct = convert_strF_to_arrF(ct_)
    temp_1 = np.empty(vis.OPERAND.shape[1], np.float64)
    if eval_method == "classic":
        weight = calculate_formula(ct, vis.OPERAND, temp_1)
    elif eval_method == "root":
        weight = calculate_formula_v2(ct, vis.OPERAND, temp_1)
    else: raise

    # Val1, Har1, Val2, Har2, Val3, Har3 = MixedSingleDoubleTriple(weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG)
    Val1, Har1, Val2, Har2, Val3, Har3 = args
    list_invest_1 = SingleYear_Har_Invest(weight, vis.BOOL_ARG, Val1)
    list_invest_2 = DoubleYear_Har_Invest(weight, vis.INDEX, vis.SYMBOL, vis.BOOL_ARG, Val2)
    list_invest_3 = TripleYear_Har_Invest(weight, vis.INDEX, vis.SYMBOL, vis.BOOL_ARG, Val3)

    #
    for i in range(vis.INDEX.shape[0]-1):
        start, end = vis.INDEX[i], vis.INDEX[i+1]
        if method == "Val1":
            wgt = list(weight[start:end]) + [Val1]
        elif method == "Val2":
            wgt = list(weight[start:end]) + [Val2]
        elif method == "Val3":
            wgt = list(weight[start:end]) + [Val3]
        else: raise ValueError(f"Method {method} is not supported")
        ranks = rankdata(np.array(wgt), method="min") - 1
        sum_rank[start:end] += ranks[:-1]
        sum_rank_ni[i] += ranks[-1]

    start, end = vis.INDEX[0], vis.INDEX[1]

    # list_invest_1
    invest = np.where(list_invest_1[start:end])[0]
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

    # list_invest_2
    invest = np.where(list_invest_2[start:end])[0]
    w_ = weight[invest]
    arg = np.argsort(w_, kind="stable")[::-1]
    invest = invest[arg] + start
    w_ = w_[arg]
    Cty2 = "_".join(map(lambda x: vis.symbol_name[vis.SYMBOL[x]], invest))
    if Cty2 == "":
        Pro2 = vis.INTEREST
    else:
        Pro2 = np.mean(vis.PROFIT[invest])
    Values_2 = "_".join(map(str, w_[arg]))

    # list_invest_3
    invest = np.where(list_invest_3[start:end])[0]
    w_ = weight[invest]
    arg = np.argsort(w_, kind="stable")[::-1]
    invest = invest[arg] + start
    w_ = w_[arg]
    Cty3 = "_".join(map(lambda x: vis.symbol_name[vis.SYMBOL[x]], invest))
    if Cty3 == "":
        Pro3 = vis.INTEREST
    else:
        Pro3 = np.mean(vis.PROFIT[invest])
    Values_3 = "_".join(map(str, w_[arg]))

    #
    slope, intercept = LinRegressPreProfit(vis, weight)

    return {
        "CT": ct_,
        "ValHar1": Val1,
        "HarNgn1": Har1,
        "CtyNgn1": Cty1,
        "Values1": Values_1,
        "ProNgn1": Pro1,
        "ValHar2": Val2,
        "HarNgn2": Har2,
        "CtyNgn2": Cty2,
        "Values2": Values_2,
        "ProNgn2": Pro2,
        "ValHar3": Val3,
        "HarNgn3": Har3,
        "CtyNgn3": Cty3,
        "Values3": Values_3,
        "ProNgn3": Pro3,
        "Slope": slope,
        "Intercept": intercept,
        "MixedHarNgn123": 3.0 / (1.0 / Har1 + 1.0 / Har2 + 1.0 / Har3),
    }


def get_dfs(vis: Base, list_ct, method: str, eval_method: str):
    list_data = []
    sum_rank = np.zeros(vis.data.shape[0])
    sum_rank_ni = np.zeros(vis.INDEX.shape[0]-1)
    for i in tqdm(range(len(list_ct))):
        ct_ = list_ct[i][0]
        args = list_ct[i][1:]
        info = get_info(vis, ct_, method, sum_rank, sum_rank_ni, eval_method, args)
        list_data.append(info)

    list_syms = []
    list_value_rank = []
    list_year = []

    for i in range(vis.INDEX.shape[0]-1):
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