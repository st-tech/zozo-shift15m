import json
import datetime
import numpy as np
import itertools

y_min = 2010
y_max = 2020

outfits = json.load(open("iqon_outfits.json"))
likes = [int(t["like_num"]) for t in outfits]
years = [
    datetime.datetime.strptime(t["publish_date"], "%Y-%m-%d").year for t in outfits
]
item_years = [
    [datetime.datetime.strptime(t["publish_date"], "%Y-%m-%d").year] * len(t["items"])
    for t in outfits
]
item_years = list(itertools.chain.from_iterable(item_years))
item_ids = [i["item_id"] for t in outfits for i in t["items"]]
item_parent_categories = [i["category_id1"] for t in outfits for i in t["items"]]
item_child_categories = [i["category_id2"] for t in outfits for i in t["items"]]
item_prices = [int(i["price"]) for t in outfits for i in t["items"]]
print("Total")
print("The number of outfits: " + str(len(outfits)))
print("The number of items: " + str(len(item_ids)))
print("The number of individual items: " + str(len(set(item_ids))))
print("The average number of likes: " + str(np.mean(likes)))
print("The median number of likes: " + str(np.median(likes)))
print("The average prices: " + str(np.mean(item_prices)))
print("The median prices: " + str(np.median(item_prices)))

for y in range(y_min, y_max + 1):
    ind = np.array(years) == y
    this_like = np.array(likes)[ind]
    item_ind = np.array(item_years) == y
    this_item_ids = np.array(item_ids)[item_ind].tolist()
    this_item_prices = np.array(item_prices)[item_ind]

    print("Year: " + str(y))
    print("The number of outfits: " + str(sum(ind)))
    print("The number of items: " + str(len(this_item_ids)))
    print("The number of individual items: " + str(len(set(this_item_ids))))
    print("The average number of likes: " + str(np.mean(this_like)))
    print("The median number of likes: " + str(np.median(this_like)))
    print("The average prices: " + str(np.mean(this_item_prices)))
    print("The median prices: " + str(np.median(this_item_prices)))
