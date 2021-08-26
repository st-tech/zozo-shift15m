import itertools
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import shift15m.constants as C


def plot_score_matrix(results: List[Dict], save_path: str):
    cm = np.zeros((len(C.YEAES), len(C.YEAES)))
    years_rev = {y: i for i, y in enumerate(C.YEAES)}
    for res in results:
        i, j = years_rev[res["train_year"]], years_rev[res["test_year"]]
        cm[j, i] = res["test_acc"]
        cm[i, i] += res["train_acc"]

    for i in range(len(C.YEAES)):
        cm[i, i] /= len(C.YEAES) - 1

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(C.YEAES))
    plt.xticks(tick_marks, C.YEAES, rotation=45)
    plt.yticks(tick_marks, C.YEAES)

    thresh = cm.max() / 1.1
    for y1, y2 in itertools.product(C.YEAES, C.YEAES):
        i, j = years_rev[y1], years_rev[y2]
        plt.text(
            i,
            j,
            f"{cm[j, i]:.1f}",
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[j, i] > thresh else "gray",
            size=8,
        )

    plt.ylabel("testing year")
    plt.xlabel("training year")
    plt.tight_layout()
    plt.savefig(save_path, orientation="portrait", pad_inches=0.1)
    plt.close()
