import chainer
import chainer.functions as F
import chainer.links as L

from networks.set_matching import SetMatchingPredictor

linear_init = chainer.initializers.LeCunUniform(scale=0.1)


class CNN(chainer.Chain):
    def __init__(self, n_units, use_fc_initializer=False):
        super(CNN, self).__init__()
        with self.init_scope():
            if use_fc_initializer == False:
                self.fc = L.Linear(4096, n_units, nobias=False)
            else:
                self.fc = L.Linear(4096, n_units, nobias=False, initialW=linear_init)

    def __call__(self, x):
        return self.fc(x)


class TwoLayeredCNN(chainer.Chain):
    def __init__(
        self, n_units,
    ):
        super(TwoLayeredCNN, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(4096, n_units, nobias=False)
            self.fc2 = L.Linear(n_units, 1, nobias=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        score = F.squeeze(self.fc2(h))
        prob = F.sigmoid(score)
        return prob


class SetMatching(chainer.Chain):
    def __init__(
        self,
        n_units,
        n_encoder_layer=2,
        n_decoder_layer=2,
        h=8,
        n_iterative=2,
        enc_apply_ln=True,
        dec_apply_ln=True,
        dec_component="MAB",
        loss="k-pair-set",
    ):

        super(SetMatching, self).__init__()
        with self.init_scope():
            self.predictor = SetMatchingPredictor(
                CNN(n_units),
                n_units,
                n_encoder_layer=n_encoder_layer,
                n_decoder_layer=n_decoder_layer,
                h=h,
                n_iterative=n_iterative,
                enc_apply_ln=enc_apply_ln,
                dec_apply_ln=dec_apply_ln,
                dec_component=dec_component,
            )
        if loss == "k-pair-set":
            self.forward = self.forward_kpair
        else:
            raise ValueError("unknown loss function.")

    def __call__(self, *args):
        return self.forward(*args)

    def forward_kpair(self, x, x_seq, y, y_seq):
        xp = self.xp
        batch = x.shape[0]

        score = self.predictor(x, x_seq, y, y_seq)

        label = xp.arange(batch, dtype=xp.int32)
        loss = F.softmax_cross_entropy(score, label)
        acc = F.accuracy(score, label)

        chainer.report(
            {"loss": loss, "acc": acc,}, self,
        )

        return loss

    def predict(self, q_imgs, q_ids, a_imgs, a_ids):
        xp = self.xp
        n_x_items = q_imgs.shape[1]
        batch = a_imgs.shape[1]

        x = F.broadcast_to(q_imgs, (batch,) + q_imgs.shape[1:])
        # (1, 8, 3, 299, 299) -> (8, 8, 3, 299, 299)
        y = a_imgs.squeeze(axis=0)
        # (1, 8, 8, 3, 299, 299) -> (8, 8, 3, 299, 299)
        x_seq = xp.broadcast_to(q_ids, (batch, n_x_items))
        y_seq = a_ids[0]
        # -> (8, 8)

        with chainer.using_config("train", False):
            score = self.predictor(x, x_seq, y, y_seq)

        return F.softmax(score[0, :], axis=0)


class SetMatchingCov(SetMatching):
    def __init__(
        self,
        n_units,
        n_encoder_layer=2,
        n_decoder_layer=2,
        h=8,
        n_iterative=2,
        enc_apply_ln=True,
        dec_apply_ln=True,
        dec_component="MAB",
        loss="k-pair-set",
        weight="mean",
        logits=True,
    ):

        super(SetMatchingCov, self).__init__(
            n_units,
            n_encoder_layer,
            n_decoder_layer,
            h,
            n_iterative,
            enc_apply_ln,
            dec_apply_ln,
            dec_component,
            loss,
        )
        with self.init_scope():
            self.weight_estimator = TwoLayeredCNN(n_units)
        self.weight = weight
        if loss == "k-pair-set":
            self.forward = self.forward_kpair
        else:
            raise ValueError("unknown loss function.")

        if weight == "mean":
            print("mean-weight selected.")
            self.calc_prob = self.mean_prob
        elif weight == "max":
            print("max-weight selected.")
            self.calc_prob = self.max_prob
        else:
            raise ValueError("unknown prob function.")

        if logits == True:
            print("logits selected.")
            self.importance = self.importance_logit
        elif logits == False:
            print("non-logit selected.")
            self.importance = self.importance_prob
        else:
            raise ValueError("unknown logits parameter.")

    def __call__(self, *args):
        return self.forward(*args)

    def importance_logit(self, prob):
        # prob = p(train|x) = 1 - p(test|x)
        xp = self.xp
        return xp.exp(1 - prob)

    def importance_prob(self, prob):
        # prob = p(train|x) = 1 - p(test|x)
        xp = self.xp
        prob = xp.where(prob == 0, xp.ones_like(prob) * 0.01, prob)
        return (1 - prob) / prob

    def estimate_weight(self, x):
        batch, n_items = x.shape[:2]
        x = F.reshape(x, (batch * n_items, x.shape[2]))
        w = self.weight_estimator(x)  # (batch*n_items,)
        w = F.reshape(w, (batch, n_items))
        return w

    def mean_prob(self, w, seq):
        xp = self.xp
        w = xp.where(seq < 0, xp.zeros_like(w.data), w.data)
        w_sum = xp.sum(w, axis=1)
        w_c = xp.sum(seq >= 0, axis=1)
        w_mean = w_sum / w_c
        return w_mean

    def max_prob(self, w, seq):
        xp = self.xp
        w = xp.where(seq < 0, xp.full(w.shape, xp.inf), w.data)
        w_min = xp.min(w, axis=1)
        # Note: here w is the probability of p(train|x), and thus it requires applying the min function
        # to calculate the maximum probability of p(test|x).
        return w_min

    def forward_kpair(self, x, x_seq, y, y_seq):
        xp = self.xp
        batch = x.shape[0]

        score = self.predictor(x, x_seq, y, y_seq)
        label = xp.arange(batch, dtype=xp.int32)
        acc = F.accuracy(score, label)
        loss = F.softmax_cross_entropy(score, label, normalize=False, reduce="no")

        # apply weights
        wx = self.estimate_weight(
            x
        )  # estimate whether x belongs to training data or not
        wy = self.estimate_weight(
            y
        )  # estimate whether y belongs to training data or not
        w = F.concat((wx, wy), axis=1)
        seq = xp.concatenate((x_seq, y_seq), axis=1)
        prob = self.calc_prob(w, seq)
        importance = self.importance(prob)
        loss = F.mean(loss * importance)

        chainer.report(
            {"loss": loss, "acc": acc,}, self,
        )

        return loss

    def predict(self, q_imgs, q_ids, a_imgs, a_ids):
        xp = self.xp
        n_x_items = q_imgs.shape[1]
        batch = a_imgs.shape[1]

        x = F.broadcast_to(q_imgs, (batch,) + q_imgs.shape[1:])
        # (1, 8, 3, 299, 299) -> (8, 8, 3, 299, 299)
        y = a_imgs.squeeze(axis=0)
        # (1, 8, 8, 3, 299, 299) -> (8, 8, 3, 299, 299)
        x_seq = xp.broadcast_to(q_ids, (batch, n_x_items))
        y_seq = a_ids[0]
        # -> (8, 8)

        with chainer.using_config("train", False):
            score = self.predictor(x, x_seq, y, y_seq)

        return F.softmax(score[0, :], axis=0)

    def load_weight(self, path):
        chainer.serializers.load_npz(path, self.weight_estimator)
        print("use the pretrained weight_estimator model.")
