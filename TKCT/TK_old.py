import numpy as np
import numba as nb
from PySources.base import decode_formula, convert_arrF_to_strF
import sqlite3
from tqdm import tqdm


@nb.njit
def is_similar(f1, f2, level):
    len1 = len(f1)
    len2 = len(f2)
    if len1 == len2:
        if np.count_nonzero(f1 != f2) >= level:
            return False
    else:
        min_ = 1000
        if len1 < len2:
            lenm = len1
            F1 = f1
            F2 = f2
        else:
            lenm = len2
            F1 = f2
            F2 = f1

        for i in range(abs(len1-len2)+1):
            temp = np.count_nonzero(F1 != F2[i:i+lenm])
            if temp < min_:
                min_ = temp
        if min_ >= level - 1:
            return False
    return True


def filter(DB_PATH, NAM_ID, target, num_field, level, FOLDER_SAVE, critical_col, filter_1525: bool):
    connect = sqlite3.connect(DB_PATH)
    cursor = connect.cursor()

    TB_NAME = f"TT{NAM_ID}" if filter_1525 else f"T{NAM_ID}"
    cursor.execute(f"SELECT count(*) FROM {TB_NAME}")
    num_rows = cursor.fetchall()[0][0]
    print(num_rows)
    list_ct = []

    cursor.execute(f"SELECT * FROM {TB_NAME} ORDER BY {critical_col} DESC;")
    # with tqdm(total=target) as pbar:
    if True:
        for k in range(num_rows):
            data = cursor.fetchone()
            ct = np.array(list(map(int, data[1].split("_"))))

            check = True
            for ct_save in list_ct[::-1]:
                if is_similar(ct, ct_save, level):
                    check = False
                    break

            if check:
                list_ct.append(ct)
                # pbar.update(1)
                if len(list_ct) == target:
                    break
            # pbar.set_postfix(processed=k+1)

    with open(f"{FOLDER_SAVE}/{NAM_ID}.txt", "w") as f:
        list_ct_str = map(lambda x: convert_arrF_to_strF(decode_formula(x, num_field)), list_ct)
        f.write("\n".join(list_ct_str))

    connect.close()
