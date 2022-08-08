import chainer
import chainer.functions as F
import chainer.links as L


class TwoLayeredCNN(chainer.Chain):
    def __init__(
        self, n_units,
    ):
        super(TwoLayeredCNN, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(4096, n_units, nobias=False)
            self.fc2 = L.Linear(n_units, 1, nobias=False)

    def __call__(self, x, label):
        return self.forward(x, label)

    def forward(self, x, label):
        h = F.relu(self.fc1(x))
        score = F.squeeze(self.fc2(h))
        loss = F.sigmoid_cross_entropy(score, label)
        acc = F.binary_accuracy(score, label)

        chainer.report(
            {"loss": loss, "acc": acc,}, self,
        )

        return loss
