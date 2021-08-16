from shift15m.datasets.numlikes_tabular import NumLikesRegression
from shift15m.datasets.sumprices_tabular import SumPricesRegression

try:
    from shift15m.datasets.imagefeature_torch import ImageFeatureDataset
    from shift15m.datasets.imagefeature_torch import (
        get_loader as get_imagefeature_dataloader,
    )

    assert ImageFeatureDataset is not None
    assert get_imagefeature_dataloader is not None
except ModuleNotFoundError:
    pass

assert NumLikesRegression is not None
assert SumPricesRegression is not None
