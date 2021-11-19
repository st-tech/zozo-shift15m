from shift15m.datasets.numlikes_tabular import NumLikesRegression
from shift15m.datasets.sumprices_tabular import SumPricesRegression

try:
    from shift15m.datasets.imagefeature_torch import ImageFeatureDataset
    from shift15m.datasets.imagefeature_torch import (
        get_loader as get_imagefeature_dataloader,
    )
    from shift15m.datasets.outfitfeature import MultisetSplitDataset

    assert ImageFeatureDataset is not None
    assert get_imagefeature_dataloader is not None
    assert MultisetSplitDataset is not None
except ModuleNotFoundError:
    pass

assert NumLikesRegression is not None
assert SumPricesRegression is not None
