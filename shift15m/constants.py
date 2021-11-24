import pathlib

repo_root = pathlib.Path(".")

BASE_URL = "https://research.zozo.com/data_release/shift15m"
OUTFIT_JSON_URL = (
    "https://research.zozo.com/data_release/shift15m/label/iqon_outfits.json"
)
JSONL = "jsonl"
PICKLE = "pickle"
ROOT = str(repo_root / "data")
DOWNLOAD = "download"
FEATURE_ROOT = str(repo_root / "data/features")
RECORDS = "records"
CATEGORY_ID_MAX = 20
YEAES = list(map(str, range(2010, 2021)))
CATEGORIES = "10,11,12,13,14,15,16".split(",")
# fmt: off
SUB_CATEGORIES = [
    "10001", "10002", "10003", "10004", "10005", "10006", "10007",
    "11001", "11002", "11003", "11004", "11005", "11006", "11007", "11008",
    "12001", "12002", "12003", "12004", "12005",
    "13001", "13002", "13003", "13004", "13005",
    "14001", "14002", "14003", "14004", "14005", "14006", "14007",
    "15001", "15002", "15003", "15004", "15005", "15006", "15007",
    "16001", "16002", "16003", "16004",
]
# fmt: on

FILES_NUM_LIKES_REGRESSION = [
    "xy_00.pickle",
    "xy_01.pickle",
    "xy_02.pickle",
    "xy_03.pickle",
    "xy_04.pickle",
    "xy_05.pickle",
    "xy_06.pickle",
    "xy_07.pickle",
    "xy_08.pickle",
    "xy_09.pickle",
]

FILES_SUM_PRICES_REGRESSION = [
    "sumprices.pickle",
]


class Keys:
    USER = "user"
    USER_ID = "user_id"
    ITEMS = "items"
    PRICE = "price"
    PUBLISH_DATE = "publish_date"
    ITEM_ID = "item_id"
    CATEGORY_ID = "category_id1"
    SUBCATEGORY_ID = "category_id2"


class ItemCatalog:
    DEFAULT_FILE = "item_catalog.txt"
    YEAR = "year"


class Tasks:
    NUM_LIKES_REGRESSION = "num_likes_regression"
    SUM_PRICES_REGRESSION = "sum_prices_regression"
