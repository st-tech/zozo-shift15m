import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import Parameter

linear_init = chainer.initializers.LeCunUniform()  # default
matrix_init = chainer.initializers._get_initializer(None)
glotuniform = chainer.initializers.GlorotUniform


def _seq_func(func, x, reconstruct_shape=True):
    """Apply a given fuction for array of ndim 3,
    shape (batch, dimension, sentence_length), channel is 2nd dim
    instead for array of ndim 2.
    """

    batch, units, length = x.shape
    # transpose to move the channel to last dim.
    e = F.transpose(x, (0, 2, 1)).reshape(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = F.transpose(e.reshape((batch, length, out_units)), (0, 2, 1))
    assert e.shape == (batch, out_units, length)
    return e


class LayerNormalizationSentence(L.LayerNormalization):
    """Position-wise Linear Layer for Sentence Block
    Position-wise layer-normalization layer for array of shape
    (batch, dimention, sentence_length)
    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = _seq_func(super(LayerNormalizationSentence, self).__call__, x)
        return y


class ConvolutionSentence(L.Convolution2D):
    """Position-wize Linear Layer for Sentence Block
    Position-wise linear layer for array of shape
    (batch, dimension, sentence_length) can be implemented a conv layer.
    """

    def __init__(self, in_channels, out_channels, nobias):
        super(ConvolutionSentence, self).__init__(
            in_channels,
            out_channels,
            ksize=1,
            stride=1,
            pad=0,
            nobias=nobias,
            initialW=linear_init,
            initial_bias=None,
        )

    def __call__(self, x):
        """Applies the linear layer.
        Args:
            x: Batch of input vector block. Its shape is
               (batchsize, in_channels, sentence_length).
        Returns:
            y: Output of the linear layer. Its shape is
               (batchsize, out_channels, sentence_length).
        """
        x = F.expand_dims(x, axis=3)
        y = super(ConvolutionSentence, self).__call__(x)
        y = F.squeeze(y, axis=3)
        return y


class MultiHeadAttention(chainer.Chain):
    """Multi-head Attention Layer for Sentence Blocks
    for efficiency, dot product to calculate query-key score
    is performed all heads together.
    """

    def __init__(self, n_units, h=8, self_attention=True, activation_fn="relu"):
        super(MultiHeadAttention, self).__init__()
        with self.init_scope():
            if self_attention:
                self.w_QKV = ConvolutionSentence(n_units, n_units * 3, nobias=True)
            else:
                self.w_Q = ConvolutionSentence(n_units, n_units, nobias=True)
                self.w_KV = ConvolutionSentence(n_units, n_units * 2, nobias=True)
            self.finishing_linear_layer = ConvolutionSentence(
                n_units, n_units, nobias=True
            )
        self.h = h
        self.scale_score = 1.0 / (n_units // h) ** 0.5
        self.is_self_attention = self_attention
        if activation_fn == "softmax":
            self.activation = self._softmax_activation
        elif activation_fn == "relu":
            self.activation = self._relu_activation
        else:
            raise ValueError("unknown activation fn.")

    def __call__(self, x, z=None, mask=None):
        xp = self.xp
        h = self.h

        if self.is_self_attention:
            Q, K, V = F.split_axis(self.w_QKV(x), 3, axis=1)
        else:
            Q = self.w_Q(x)
            K, V = F.split_axis(self.w_KV(z), 2, axis=1)
        batch, n_units, n_queries = Q.shape
        _, _, n_keys = K.shape

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Per form Multi-head Attention using pseudo batching all together
        # at once for efficiency.
        batch_Q = F.concat(F.split_axis(Q, h, axis=1), axis=0)
        batch_K = F.concat(F.split_axis(K, h, axis=1), axis=0)
        batch_V = F.concat(F.split_axis(V, h, axis=1), axis=0)
        assert batch_Q.shape == (batch * h, n_units // h, n_queries)
        assert batch_K.shape == (batch * h, n_units // h, n_keys)
        assert batch_V.shape == (batch * h, n_units // h, n_keys)

        batch_A = F.batch_matmul(batch_Q, batch_K, transa=True)
        batch_A = self.activation(batch_A, mask, h, batch, n_queries, n_keys, xp)
        assert batch_A.shape == (batch * h, n_queries, n_keys)

        # Calculate weighted sum.
        batch_A, batch_V = F.broadcast(batch_A[:, None], batch_V[:, :, None])
        batch_C = F.sum(batch_A * batch_V, axis=3)
        assert batch_C.shape == (batch * h, n_units // h, n_queries)
        C = F.concat(F.split_axis(batch_C, h, axis=0), axis=1)
        assert C.shape == (batch, n_units, n_queries)
        C = self.finishing_linear_layer(C)
        return C

    def _softmax_activation(self, _batch_A, _mask, h, batch, n_queries, n_keys, xp):
        mask = xp.concatenate([_mask] * h, axis=0)

        batch_A = _batch_A * self.scale_score
        batch_A = F.where(mask, batch_A, xp.full(batch_A.shape, -np.inf, "f"))
        batch_A = F.softmax(batch_A, axis=2)
        batch_A = F.where(xp.isnan(batch_A.data), xp.zeros(batch_A.shape, "f"), batch_A)
        return batch_A

    def _relu_activation(self, _batch_A, _mask, h, batch, n_queries, n_keys, xp):
        m = np.repeat(_mask.sum(2), n_keys).reshape(batch, n_queries, n_keys)
        n_elements = xp.where(_mask, m, xp.ones_like(_mask))
        n_elements = xp.concatenate([n_elements] * h, axis=0)
        mask = xp.concatenate([_mask] * h, axis=0)

        batch_A = F.relu(_batch_A)
        batch_A *= self.scale_score
        batch_A = F.where(mask, batch_A, xp.full(batch_A.shape, -np.inf, "f"))
        batch_A /= n_elements
        batch_A = F.where(xp.isinf(batch_A.data), xp.zeros(batch_A.shape, "f"), batch_A)
        return batch_A

    def get_attnmap(self, x, z=None, mask=None):
        xp = self.xp
        h = self.h

        if self.is_self_attention:
            Q, K, V = F.split_axis(self.w_QKV(x), 3, axis=1)
        else:
            Q = self.w_Q(x)
            K, V = F.split_axis(self.w_KV(z), 2, axis=1)
        batch, n_units, n_queries = Q.shape
        _, _, n_keys = K.shape

        batch_Q = F.concat(F.split_axis(Q, h, axis=1), axis=0)
        batch_K = F.concat(F.split_axis(K, h, axis=1), axis=0)

        batch_A = F.batch_matmul(batch_Q, batch_K, transa=True)
        return self.activation(batch_A, mask, h, batch, n_queries, n_keys, xp)


class MultiHeadSimilarity(chainer.Chain):
    def __init__(self, n_units, h=8):
        super(MultiHeadSimilarity, self).__init__()
        with self.init_scope():
            self.w_Q = ConvolutionSentence(n_units, n_units, nobias=True)
            self.w_K = ConvolutionSentence(n_units, n_units, nobias=True)
            self.ln = LayerNormalizationSentence(n_units, eps=1e-6)
            if h > 1:
                self.finishing_linear_layer = ConvolutionSentence(
                    n_units, n_units, nobias=True
                )
            else:
                self.finishing_linear_layer = F.identity
        self.h = h
        self.scale_score = 1.0 / (n_units // h) ** 0.5

    def __call__(self, x, z=None, mask=None):
        # This function calculates:
        #   x_i = LN(0.5*(q(x_i) + (1/n_y) Σ_j ReLU(q(x_i)^T k(y_j))k(y_j))), where j=[1, ..., n_y].
        # The matrix representation is:
        #  X = LN(X + (1/y_counts)ReLU(QK^T)V)
        # mask: (batch, n_x, n_z)
        xp = self.xp
        h = self.h

        Q = self.w_Q(x)
        K = self.w_K(z)
        batch, n_units, n_queries = Q.shape
        _, _, n_keys = K.shape

        n_elements = xp.sum(mask, axis=2)
        n_elements = xp.where(
            n_elements == 0, xp.ones(n_elements.shape, "f"), n_elements
        )

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Per form Multi-head Attention using pseudo batching all together
        # at once for efficiency.
        batch_Q = F.concat(F.split_axis(Q, h, axis=1), axis=0)
        batch_K = F.concat(F.split_axis(K, h, axis=1), axis=0)
        assert batch_Q.shape == (batch * h, n_units // h, n_queries)
        assert batch_K.shape == (batch * h, n_units // h, n_keys)

        mask = xp.concatenate([mask] * h, axis=0)
        batch_A = F.batch_matmul(batch_Q, batch_K, transa=True)
        batch_A = F.relu(batch_A)
        # ^ we expect that this op will provide a rich representation.
        # see https://st-tech.slack.com/archives/CA0SRHG85/p1550299374051100
        batch_A *= self.scale_score
        batch_A = F.where(mask, batch_A, xp.zeros(batch_A.shape, "f"))
        assert batch_A.shape == (batch * h, n_queries, n_keys)
        batch_C = F.matmul(batch_A, batch_K, transb=True)
        assert batch_C.shape == (batch * h, n_queries, n_units // h)
        C = F.concat(F.split_axis(batch_C, h, axis=0), axis=2)
        C = F.transpose(C, (0, 2, 1))
        assert C.shape == (batch, n_units, n_queries)
        C /= n_elements[:, None, :]

        E = 0.5 * (Q + C)
        E = self.finishing_linear_layer(E)
        return E

    def get_attnmap(self, x, z=None, mask=None):
        xp = self.xp
        h = self.h

        Q = self.w_Q(x)
        K = self.w_K(z)

        batch_Q = F.concat(F.split_axis(Q, h, axis=1), axis=0)
        batch_K = F.concat(F.split_axis(K, h, axis=1), axis=0)

        mask = xp.concatenate([mask] * h, axis=0)
        batch_A = F.batch_matmul(batch_Q, batch_K, transa=True)
        batch_A = F.relu(batch_A)
        batch_A *= self.scale_score
        batch_A = F.where(mask, batch_A, xp.zeros(batch_A.shape, "f"))
        return batch_A


class MultiHeadExpectation(chainer.Chain):
    def __init__(self, n_units, h=8, actibation_fn="relu"):
        super(MultiHeadExpectation, self).__init__()
        with self.init_scope():
            self.w = ConvolutionSentence(n_units, n_units, nobias=True)
            if h != 1:
                self.fc = L.Linear(h, 1)
            else:
                self.fc = F.identity
        self.h = h
        self.scale_score = 1.0 / (n_units // h) ** 0.5
        self.actibation_fn = actibation_fn

        # for ablation study
        if actibation_fn.lower() == "relu":
            self.act_func = F.relu
            # ^ we expect that this op will provide a rich representation.
            # see https://st-tech.slack.com/archives/CA0SRHG85/p1550299374051100
        elif actibation_fn.lower() == "none":
            self.act_func = F.identity
        else:
            raise ValueError("no definition for cs_actibation_fn.")

    def __call__(self, x, y, yx_mask):
        # yx_mask: (batch, n_x, n_y)
        xp = self.xp
        h = self.h

        batch, n_units, n_elem_x = x.shape
        _, _, n_elem_y = y.shape

        _x = self.w(x)
        _y = self.w(y)

        n_elements = xp.sum(yx_mask, axis=(1, 2))
        n_elements = xp.where(
            n_elements == 0, xp.ones(n_elements.shape, "f"), n_elements
        )

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Per form Multi-head Attention using pseudo batching all together
        # at once for efficiency.
        batch_x = F.concat(F.split_axis(_x, h, axis=1), axis=0)
        batch_y = F.concat(F.split_axis(_y, h, axis=1), axis=0)
        assert batch_x.shape == (batch * h, n_units // h, n_elem_x)
        assert batch_y.shape == (batch * h, n_units // h, n_elem_y)
        mask = xp.concatenate([yx_mask] * h, axis=0)
        batch_xy = F.batch_matmul(batch_x, batch_y, transa=True)

        batch_xy = self.act_func(batch_xy)
        batch_xy *= (
            self.scale_score
        )  # if self.act_func is nonlinear, you should consider moving this line to before self.act_func
        batch_xy = F.where(mask, batch_xy, xp.full(batch_xy.shape, 0, "f"))

        xy = F.concat(F.split_axis(batch_xy[:, :, :, None], h, axis=0), axis=3)

        similarity = F.squeeze(F.sum(xy, axis=(1, 2)))

        # for ablation study
        if h != 1:
            n_elements = n_elements[:, None]
        else:
            n_elements = n_elements
        expectation = similarity / n_elements
        expectation = self.fc(expectation)

        return expectation


class FeedForwardLayer(chainer.Chain):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4
        with self.init_scope():
            self.w_1 = ConvolutionSentence(None, n_inner_units, nobias=False)
            self.w_2 = ConvolutionSentence(n_inner_units, n_units, nobias=False)

    def __call__(self, e):
        e = self.w_1(e)
        e = F.leaky_relu(e)  # in the original paper, this function is a relu.
        e = self.w_2(e)
        return e


class SAB(chainer.Chain):
    def __init__(self, n_units, h=8, apply_ln=True):
        super(SAB, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(
                n_units, h, activation_fn="softmax"
            )
            self.feed_forward = FeedForwardLayer(n_units)
            if apply_ln:
                self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
                self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
            else:
                self.ln_1 = F.identity
                self.ln_2 = F.identity

    def __call__(self, e, xx_mask):
        sub = self.self_attention(e, mask=xx_mask)
        e = e + sub
        e = self.ln_1(e)
        sub = self.feed_forward(e)
        e = e + sub
        e = self.ln_2(e)
        return e

    def get_attnmap(self, x, xx_mask):
        attn = self.self_attention.get_attnmap(x, mask=xx_mask)
        # attn1: (8*196, 8*196)
        return attn


class ISAB(chainer.Chain):
    def __init__(self, n_units, h=8, m=16):
        super(ISAB, self).__init__()
        with self.init_scope():
            self.I = Parameter(matrix_init, (n_units, m))
            self.self_attention_1 = MultiHeadAttention(n_units, h, self_attention=False)
            self.feed_forward_1 = FeedForwardLayer(n_units)
            self.ln_1_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_1_2 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.self_attention_2 = MultiHeadAttention(n_units, h, self_attention=False)
            self.feed_forward_2 = FeedForwardLayer(n_units)
            self.ln_2_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.m = m

    def __call__(self, x, xi_mask, ix_mask):
        # ISAB(X) = MAB(X, H) \in R^{n \times d}
        # where H = MAB(I, X) \ in R^{m \times d}
        # MAB(u, v) = LayerNorm(H + rFF(H))
        # where H = LayerNorm(u + Multihead(u, v, v))
        batch, n_units, _ = x.shape
        # MAB(I, X), mask: m -> n
        i = F.broadcast_to(self.I, (batch, n_units, self.m))
        mh_ix = self.self_attention_1(i, x, mask=xi_mask)
        h = self.ln_1_1(i + mh_ix)
        rff = self.feed_forward_1(h)
        h = self.ln_1_2(h + rff)
        # MAB(X, H), mask: n -> m
        mh_xh = self.self_attention_2(x, h, mask=ix_mask)
        h = self.ln_2_1(x + mh_xh)
        rff = self.feed_forward_2(h)
        h = self.ln_2_2(h + rff)
        return h

    def get_attnmap(self, x, xi_mask, ix_mask):
        batch, n_units, _ = x.shape
        # MAB(I, X), mask: m -> n
        i = F.broadcast_to(self.I, (batch, n_units, self.m))
        attn1 = self.self_attention_1.get_attnmap(i, x, mask=xi_mask)
        # attn1: (16, 8*196)

        mh_ix = self.self_attention_1(i, x, mask=xi_mask)
        h = self.ln_1_1(i + mh_ix)
        rff = self.feed_forward_1(h)
        h = self.ln_1_2(h + rff)
        attn2 = self.self_attention_2.get_attnmap(x, h, mask=ix_mask)
        # attn2: (8*196, 16)
        psudo_attn = F.batch_matmul(attn2, attn1)
        return psudo_attn


class MAB(chainer.Chain):
    """
    MAB(X, Y) = LN(H + rFF(H))
    where H = LN(X + Attention(X, Y, Y))
    """

    def __init__(self, n_units, h=8, activation_fn="relu", apply_ln=True):
        super(MAB, self).__init__()
        with self.init_scope():
            self.attention = MultiHeadAttention(
                n_units, h, self_attention=False, activation_fn=activation_fn
            )
            self.rff = FeedForwardLayer(n_units)
            if apply_ln:
                self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
                self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
            else:
                self.ln_1 = F.identity
                self.ln_2 = F.identity

    def __call__(self, x, y, mask):
        # MAB(u, v) = LayerNorm(H + rFF(H))
        # where H = LayerNorm(u + Multihead(u, v, v))
        h = self.ln_1(x + self.attention(x, y, mask))
        e = self.ln_2(h + self.rff(h))
        return e


class PMA(chainer.Chain):
    """
    PMAk(Z) = MAB(S, rFF(Z))
            = LN(H + rFF(H))
    where H = LN(S + Multihead(S, rFF(Z), rFF(Z)))
    """

    def __init__(self, n_units, h=8, n_output_instances=None, mh_activation="softmax"):
        self.k = n_output_instances
        super(PMA, self).__init__()
        with self.init_scope():
            self.attention = MultiHeadAttention(
                n_units, h, self_attention=False, activation_fn=mh_activation
            )
            self.rff_1 = FeedForwardLayer(n_units)
            self.rff_2 = FeedForwardLayer(n_units)
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)

    def __call__(self, z, mask, s=None):
        xp = self.xp
        batch, n_units, _ = z.shape
        if s is None:
            s = xp.ones((batch, n_units, self.k), "f")
        else:
            k = s.shape[0]
            s = F.broadcast_to(F.transpose(s, (1, 0)), (batch, n_units, k))

        e = self.rff_1(z)
        h = self.ln_1(s + self.attention(s, e, mask))
        e = self.ln_2(h + self.rff_2(h))
        return e


class SetEncoder(chainer.Chain):
    def __init__(self, n_units, n_layers=2, h=8, apply_ln=True):
        super(SetEncoder, self).__init__()
        self.layer_names = list()
        for i in range(1, n_layers + 1):
            name = "sab_{}".format(i)
            layer = SAB(n_units, h, apply_ln=apply_ln)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, mask):
        for name in self.layer_names:
            e = getattr(self, name)(e, mask)
        return e

    def get_attnmap(self, e, mask):
        attnmaps = []
        for name in self.layer_names:
            attnmaps.append(getattr(self, name).get_attnmap(e, mask))
            e = getattr(self, name)(e, mask)
        return attnmaps


class SetEncoderISAB(chainer.Chain):
    def __init__(self, n_units, n_layers=2, h=8, m=16):
        super(SetEncoderISAB, self).__init__()
        self.layer_names = list()
        for i in range(1, n_layers + 1):
            name = "isab_{}".format(i)
            layer = ISAB(n_units, h, m)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, xi_mask, ix_mask):
        for name in self.layer_names:
            e = getattr(self, name)(e, xi_mask, ix_mask)
        return e

    def get_attnmap(self, e, xi_mask, ix_mask):
        attnmaps = []
        for name in self.layer_names:
            attnmaps.append(getattr(self, name).get_attnmap(e, xi_mask, ix_mask))
            e = getattr(self, name)(e, xi_mask, ix_mask)
        return attnmaps


class SetDecoderNoSAB(chainer.Chain):
    def __init__(self, n_units, h=8):
        super(SetDecoderNoSAB, self).__init__()
        with self.init_scope():
            self.rff_1 = FeedForwardLayer(n_units)
            self.mab = MultiHeadAttention(n_units, h, self_attention=False)
            self.rff_2 = FeedForwardLayer(n_units)

    def __call__(self, z, mask):
        xp = self.xp
        batch, n_units, _ = z.shape

        e = self.rff_1(z)
        e = self.mab(xp.ones((batch, n_units, 1), "f"), e, mask)
        e = self.rff_2(e)
        return e


class SetDecoderNoSAB_paramS(chainer.Chain):
    def __init__(self, n_units, h=8):
        super(SetDecoderNoSAB_paramS, self).__init__()
        with self.init_scope():
            self.S = Parameter(matrix_init, (n_units, 1))
            self.rff_1 = FeedForwardLayer(n_units)
            self.mab = MultiHeadAttention(n_units, h, self_attention=False)
            self.rff_2 = FeedForwardLayer(n_units)

    def __call__(self, z, mask):
        batch, n_units, _ = z.shape

        s = F.broadcast_to(self.S, (batch, n_units, 1))
        e = self.rff_1(z)
        e = self.mab(s, e, mask)
        e = self.rff_2(e)
        return e


class SetDecoderNoSAB_inpS(chainer.Chain):
    def __init__(self, n_units, h=8):
        super(SetDecoderNoSAB_inpS, self).__init__()
        with self.init_scope():
            self.rff_1 = FeedForwardLayer(n_units)
            self.mab = MultiHeadAttention(
                n_units, h, self_attention=False, activation_fn="softmax"
            )
            self.rff_2 = FeedForwardLayer(n_units)

    def __call__(self, z, s, mask):
        batch, n_units, _ = z.shape
        n_candidates = s.shape[0]

        # assert s.shape == (batch, n_units)
        s = F.broadcast_to(F.transpose(s, (1, 0)), (batch, n_units, n_candidates))
        e = self.rff_1(z)
        e = self.mab(s, e, mask)
        e = self.rff_2(e)
        return e


class SetRichDecoderNoSAB_paramS(chainer.Chain):
    def __init__(self, n_units, h=8, apply_last_rff=True):
        super(SetRichDecoderNoSAB_paramS, self).__init__()
        with self.init_scope():
            self.S = Parameter(matrix_init, (n_units, 1))
            self.rff_1 = FeedForwardLayer(n_units)
            self.mab = MAB(n_units, h, activation_fn="softmax")
            if apply_last_rff:
                self.rff_2 = FeedForwardLayer(n_units)
            else:
                self.rff_2 = F.identity

    def __call__(self, z, mask):
        batch, n_units, _ = z.shape

        s = F.broadcast_to(self.S, (batch, n_units, 1))
        e = self.rff_1(z)
        e = self.mab(s, e, mask)
        e = self.rff_2(e)
        return e


class SimpleSetDecoder(chainer.Chain):
    def __init__(
        self,
        n_units,
        h=8,
        apply_last_rff=True,
        n_units_last=None,
        component="MAB",
        activation_fn="relu",
        apply_ln=True,
    ):
        super(SimpleSetDecoder, self).__init__()
        with self.init_scope():
            self.rff_1 = FeedForwardLayer(n_units)
            if component == "MAB":
                self.mab = MAB(
                    n_units, h, activation_fn=activation_fn, apply_ln=apply_ln
                )
            elif component == "MHAtt":
                self.mab = MultiHeadAttention(
                    n_units, h, self_attention=False, activation_fn=activation_fn
                )
            elif component == "MHSim":
                self.mab = MultiHeadSimilarity(n_units, h)
            else:
                raise ValueError("no definition for MAB.")
            if apply_last_rff:
                assert n_units_last is not None
                self.rff_2 = FeedForwardLayer(n_units_last)
            else:
                self.rff_2 = F.identity

    def __call__(self, z, s, mask):
        # The size of s is (batch, u_units, batch), broad-casted or generated by MAB.
        # If broad-casted, the 3rd axis of s is just copied from old s: (batch, u_units).
        # In MAB generating case, s is composed of batch x batch feature vectors.
        s = self.rff_1(s)  # (batch, n_units*3, batch)
        s = self.mab(s, z, mask)  # (batch, n_units, batch)
        s = self.rff_2(s)
        return s

    def get_attnmap(self, z, s, mask):
        s = self.rff_1(s)  # (batch, n_units*3, batch)
        attnmap = self.mab.get_attnmap(s, z, mask)  # (batch, n_units, batch)
        return attnmap


class SetDecoder(chainer.Chain):
    """
    Decoder(Z) = rFF(SAB(PMAk(Z))) \in R^{k \times d}
    """

    def __init__(self, n_units, h=8, n_output_instances=1):
        super(SetDecoder, self).__init__()
        with self.init_scope():
            self.pma = PMA(n_units, h=h, n_output_instances=n_output_instances)
            self.sab = SAB(n_units, h)
            self.rff = FeedForwardLayer(n_units)

    def __call__(self, z, xy_mask, yy_mask):
        e = self.pma(z, xy_mask)
        e = self.sab(e, yy_mask)
        e = self.rff(e)
        return e


class SetDecoder_paramS(chainer.Chain):
    """
    Decoder(Z) = rFF(SAB(PMAk(Z))) \in R^{k \times d}
    """

    def __init__(self, n_units, h=8, n_output_instances=1):
        super(SetDecoder_paramS, self).__init__()
        init = glotuniform(scale=np.sqrt(6.0 / (n_units * 2)))
        with self.init_scope():
            self.S = Parameter(init, (n_output_instances, n_units))
            self.pma = PMA(n_units, h=h, n_output_instances=n_output_instances)
            self.sab = SAB(n_units, h)
            self.rff = FeedForwardLayer(n_units)

    def __call__(self, z, xy_mask, yy_mask):
        e = self.pma(z, xy_mask, self.S)
        e = self.sab(e, yy_mask)
        e = self.rff(e)
        return e


class StackedSimpleSetDecoder(chainer.Chain):
    def __init__(
        self,
        n_units,
        n_layers=2,
        h=8,
        n_units_last=None,
        component="MAB",
        activation_fn="relu",
        apply_ln=True,
    ):
        super(StackedSimpleSetDecoder, self).__init__()
        self.layer_names = list()

        # middle layers
        for i in range(1, n_layers):
            name = "dec_{}".format(i)
            layer = SimpleSetDecoder(
                n_units,
                h,
                apply_last_rff=False,
                component=component,
                activation_fn=activation_fn,
                apply_ln=apply_ln,
            )
            self.add_link(name, layer)
            self.layer_names.append(name)
        # last layer
        if n_units_last is None:
            n_units_last = n_units
        name = "dec_{}".format(n_layers)
        layer = SimpleSetDecoder(
            n_units,
            h,
            apply_last_rff=True,
            n_units_last=n_units_last,
            component=component,
            activation_fn=activation_fn,
            apply_ln=apply_ln,
        )
        self.add_link(name, layer)
        self.layer_names.append(name)

    def __call__(self, z, s, xy_mask):
        if len(s.shape) == 2:
            batch, n_units, _ = z.shape
            n_candidates = s.shape[0]
            assert s.shape[1] == n_units
            s = F.broadcast_to(F.transpose(s, (1, 0)), (batch, n_units, n_candidates))
        for name in self.layer_names:
            s = getattr(self, name)(z, s, xy_mask)
        return s

    def get_attnmap(self, z, s, xy_mask):
        attnmap = []
        if len(s.shape) == 2:
            batch, n_units, _ = z.shape
            n_candidates = s.shape[0]
            assert s.shape[1] == n_units
            s = F.broadcast_to(F.transpose(s, (1, 0)), (batch, n_units, n_candidates))
        for name in self.layer_names:
            attnmap.append(getattr(self, name).get_attnmap(z, s, xy_mask))
            s = getattr(self, name)(z, s, xy_mask)
        return attnmap
