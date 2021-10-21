from typing import Callable, Sequence, Union

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.accuracy import _BaseClassification
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class BinaryAccuracy(_BaseClassification):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(BinaryAccuracy, self).__init__(
            output_transform=output_transform,
            is_multilabel=is_multilabel,
            device=device,
        )

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super(BinaryAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        y_pred, y = output[0].detach(), output[1].detach()

        y_pred = torch.where(
            y_pred > 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred)
        )
        correct = torch.eq(y_pred.view(-1).to(y), y.view(-1))

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError(
                "Accuracy must have at least one example before it can be computed."
            )
        return self._num_correct.item() / self._num_examples
