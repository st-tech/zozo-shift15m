import json
import datetime
import numpy as np

y_min = 2010
y_max = 2020

data = json.load(open("rm_no_img_set_saito_women_likeszero_items4.json"))
item_ids = [i["item_id"] for t in data for i in t["items"]]
cats = [[i["category_id1"] for i in t["items"]] for t in data]
like = [int(t["like_num"]) for t in data]
date = [datetime.datetime.strptime(t["publish_date"], "%Y-%m-%d") for t in data]
years = [d.year for d in date]
print("Total")
print("The number of outfits: " + str(len(data)))
print("The number of items: " + str(len(item_ids)))
print("The number of individual items: " + str(len(set(item_ids))))
print("The average number of likes: " + str(np.mean(like)))
print("The median number of likes: " + str(np.median(like)))

for y in range(y_min, y_max + 1):
    ind = np.array(years) == y
    this_data = np.array(data)[ind]
    this_item_ids = [i["item_id"] for t in this_data for i in t["items"]]
    this_like = [int(t["like_num"]) for t in this_data]

    print("Year: " + str(y))
    print("The number of outfits: " + str(len(this_data)))
    print("The number of items: " + str(len(this_item_ids)))
    print("The number of individual items: " + str(len(set(this_item_ids))))
    print("The average number of likes: " + str(np.mean(this_like)))
    print("The median number of likes: " + str(np.median(this_like)))
