import torch
import torch.nn as nn
import torch.nn.functional as F
from set_matching.models.set_matching import SetMatching


class TwoLayeredLinear(nn.Module):
    def __init__(self, n_units):
        super(TwoLayeredLinear, self).__init__()
        self.fc1 = nn.Linear(4096, n_units)
        self.fc2 = nn.Linear(n_units, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        score = self.fc2(h).squeeze()
        prob = torch.sigmoid(score)
        return prob


class SetMatchingCov(SetMatching):
    def __init__(
        self,
        n_units,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_heads=8,
        n_iterative=2,
        enc_apply_ln=True,
        dec_apply_ln=True,
        dec_component="MAB",
        embedder_arch="linear",
        weight="mean",
        logits=True,
        pretrained_weight=None,
    ):

        super(SetMatchingCov, self).__init__(
            n_units,
            n_encoder_layers,
            n_decoder_layers,
            n_heads,
            n_iterative,
            enc_apply_ln,
            dec_apply_ln,
            dec_component,
            embedder_arch,
        )
        self.weight_estimator = TwoLayeredLinear(n_units)
        self.weight = weight

        if weight == "mean":
            print("mean-weight selected.")
            self.calc_prob = self.mean_prob
        elif weight == "max":
            print("max-weight selected.")
            self.calc_prob = self.max_prob
        else:
            raise ValueError("unknown prob function.")

        if logits:
            print("logits selected.")
            self.importance = self.importance_logit
        else:
            print("non-logit selected.")
            self.importance = self.importance_prob

        self.register_buffer("dummy_var", torch.empty(0))

        if pretrained_weight:
            with open(pretrained_weight, "rb") as f:
                self.weight_estimator.load_state_dict(torch.load(f))
            for param in self.weight_estimator.parameters():
                param.requires_grad = False

    def importance_logit(self, prob):
        # prob = p(train|x) = 1 - p(test|x)
        return torch.exp(1 - prob)

    def importance_prob(self, prob):
        # prob = p(train|x) = 1 - p(test|x)
        prob = torch.where(torch.eq(prob, 0), torch.ones_like(prob) * 0.01, prob)
        return (1 - prob) / prob

    def estimate_weight(self, x):
        batch, n_items = x.shape[:2]
        x = x.reshape((batch * n_items, x.shape[2]))
        w = self.weight_estimator(x)  # (batch*n_items,)
        w = w.reshape((batch, n_items))
        return w

    def mean_prob(self, w, seq):
        w = torch.where(seq < 0, torch.zeros_like(w), w)
        w_sum = w.sum(dim=1)
        w_c = (seq >= 0).sum(dim=1)
        w_mean = w_sum / w_c
        return w_mean

    def max_prob(self, w, seq):
        w = torch.where(
            seq < 0,
            torch.full(
                w.shape, float("inf"), dtype=torch.float32, device=self.dummy_var.device
            ),
            w,
        )
        w_min, _ = w.min(dim=1)
        # Note: here w is the probability of p(train|x), and thus it requires applying the min function
        # to calculate the maximum probability of p(test|x).
        return w_min

    def forward(self, x, x_seq, y, y_seq):
        score = super().forward(x, x_seq, y, y_seq)

        # apply weights
        wx = self.estimate_weight(
            x
        )  # estimate whether x belongs to training data or not
        wy = self.estimate_weight(
            y
        )  # estimate whether y belongs to training data or not
        w = torch.cat((wx, wy), dim=1)
        seq = torch.cat((x_seq, y_seq), dim=1)
        prob = self.calc_prob(w, seq)
        importance = self.importance(prob)

        return score, importance
