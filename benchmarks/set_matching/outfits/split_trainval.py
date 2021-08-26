import json
import datetime
import numpy as np
import os


output_root = 'set_matching_labels'
data = json.load(open('iqon_outfits.json'))
cats = [[i['category_id1'] for i in t['items']] for t in data]
like = [int(t['like_num']) for t in data]
date = [datetime.datetime.strptime(t['publish_date'], '%Y-%m-%d') for t in data]

years = [
    [2013, 2013],
    [2013, 2014],
    [2013, 2015],
    [2013, 2016],
    [2013, 2017],
]

train = 30816
val = 3851
test = 3851

for (ys, ye) in years:
    str_year = str(ys)+'-'+str(ye)+'-'
    time_ys0 = datetime.datetime(ys, 1, 1, 0, 0)
    time_ys1 = datetime.datetime(ys + 1, 1, 1, 0, 0)
    time_ye0 = datetime.datetime(ye, 1, 1, 0, 0)
    time_ye1 = datetime.datetime(ye + 1, 1, 1, 0, 0)
    ind_ys = [(time_ys0 <= d) and (d < time_ys1) and (l >= 50) and (len(set(c)) >= 4) for d, l, c in zip(date, like, cats)]
    ind_ye = [(time_ye0 <= d) and (d < time_ye1) and (l >= 50) and (len(set(c)) >= 4) for d, l, c in zip(date, like, cats)]
    data_ys = np.array(data)[ind_ys]
    data_ye = np.array(data)[ind_ye]

    np.random.seed(0)
    data_perm = np.random.permutation(data_ys)
    data_tr = data_perm[0:train]
    if ys != ye:
        np.random.seed(0)
        data_perm = np.random.permutation(data_ye)
        data_vl = data_perm[0:val]
        data_te = data_perm[val:val+test]
    else:
        data_vl = data_perm[train:train+val]
        data_te = data_perm[train+val:train+val+test]
    data_tr=data_tr.tolist()
    data_vl=data_vl.tolist()
    data_te=data_te.tolist()
    output_dir = os.path.join(output_root, str_year, 'label1')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(data_tr, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, 'valid.json'), 'w') as f:
        json.dump(data_vl, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(data_te, f, indent=4, ensure_ascii=True)

    np.random.seed(1)
    data_perm = np.random.permutation(data_ys)
    data_tr = data_perm[0:train]
    if ys != ye:
        np.random.seed(1)
        data_perm = np.random.permutation(data_ye)
        data_vl = data_perm[0:val]
        data_te = data_perm[val:val+test]
    else:
        data_vl = data_perm[train:train+val]
        data_te = data_perm[train+val:train+val+test]
    data_tr=data_tr.tolist()
    data_vl=data_vl.tolist()
    data_te=data_te.tolist()
    output_dir = os.path.join(output_root, str_year, 'label2')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(data_tr, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, 'valid.json'), 'w') as f:
        json.dump(data_vl, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(data_te, f, indent=4, ensure_ascii=True)

    np.random.seed(2)
    data_perm = np.random.permutation(data_ys)
    data_tr = data_perm[0:train]
    if ys != ye:
        np.random.seed(2)
        data_perm = np.random.permutation(data_ye)
        data_vl = data_perm[0:val]
        data_te = data_perm[val:val+test]
    else:
        data_vl = data_perm[train:train+val]
        data_te = data_perm[train+val:train+val+test]
    data_tr=data_tr.tolist()
    data_vl=data_vl.tolist()
    data_te=data_te.tolist()
    output_dir = os.path.join(output_root, str_year, 'label3')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(data_tr, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, 'valid.json'), 'w') as f:
        json.dump(data_vl, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(data_te, f, indent=4, ensure_ascii=True)