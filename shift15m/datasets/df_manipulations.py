import numpy as np
from shift15m import constants as C


def extract_user_id(x: dict) -> int:
    keys: dict_keys = x.keys()
    if C.Keys.USER_ID in keys:
        return int(x[C.Keys.USER_ID])
    else:
        return -1


def price_sum(x: list) -> int:
    res = 0
    for item in x:
        res += int(item[C.Keys.PRICE])

    return res


def price_mean(x: list) -> int:
    res: list = []
    for item in x:
        res.append(int(item[C.Keys.PRICE]))

    return np.mean(res) if len(res) > 0 else 0


def price_max(x: list) -> int:
    res: list = []
    for item in x:
        res.append(int(item[C.Keys.PRICE]))

    return np.max(res) if len(res) > 0 else 0


def price_min(x: list) -> int:
    res: list = []
    for item in x:
        res.append(int(item[C.Keys.PRICE]))

    return np.min(res) if len(res) > 0 else 0


def categories_count_embedding_id1(x: list) -> np.ndarray:
    return categories_count_embedding(x, key="category_id1")


def categories_count_embedding(x: list, key="category_id1") -> np.ndarray:
    categories = []
    for item in x:
        categories.append(int(item[key]))

    one_hot: np.ndarray = np.identity(C.CATEGORY_ID_MAX)[categories]
    res: np.ndarray = np.zeros(C.CATEGORY_ID_MAX)

    for a in one_hot:
        res += a

    return res
