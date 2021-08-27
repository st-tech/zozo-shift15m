import chainer
import chainer.functions as F

from networks.transformer_modules import SetEncoder
from networks.transformer_modules import StackedSimpleSetDecoder
from networks.transformer_modules import MultiHeadExpectation
from networks.utils import make_attn_mask


class SetMatchingPredictor(chainer.Chain):
    def __init__(
        self,
        embedder,
        n_units,
        n_encoder_layer=2,
        n_decoder_layer=2,
        h=8,
        n_iterative=2,
        enc_apply_ln=True,
        dec_apply_ln=True,
        dec_component="MHSim",
        cs_actibation_fn="relu",
        h_ct=False,
        h_cs=False,
    ):

        if h_ct == False:
            h_ct = h
        if h_cs == False:
            h_cs = h
        super(SetMatchingPredictor, self).__init__()
        with self.init_scope():
            self.embedder = embedder

            for i in range(0, n_iterative):
                name = "enc_{}".format(i)
                layer = SetEncoder(
                    n_units, n_layers=n_encoder_layer, h=h, apply_ln=enc_apply_ln
                )
                self.add_link(name, layer)
                name = "dec_{}".format(i)
                layer = StackedSimpleSetDecoder(
                    n_units,
                    n_layers=n_decoder_layer,
                    h=h_ct,
                    component=dec_component,
                    activation_fn="relu",
                    apply_ln=dec_apply_ln,
                )
                self.add_link(name, layer)
            self.last_dec = MultiHeadExpectation(
                n_units, h=h_cs, actibation_fn=cs_actibation_fn
            )

        self.n_units = n_units
        self.n_iterative = n_iterative

    def __call__(self, x, x_seq, y, y_seq):
        xp = self.xp
        batch = x.shape[0]
        n_x_items = x.shape[1]
        n_y_items = y.shape[1]
        n_units = self.n_units

        x = self.embed_reshape_transpose(x)
        y = self.embed_reshape_transpose(y)

        # Extract set features by the encoder
        x = F.reshape(
            F.broadcast_to(
                F.expand_dims(x, axis=1), (batch, batch, n_units, n_x_items)
            ),
            (batch * batch, n_units, n_x_items),
        )  # [x_1, x_1, ...]
        y = F.reshape(
            F.broadcast_to(
                F.expand_dims(y, axis=0), (batch, batch, n_units, n_y_items)
            ),
            (batch * batch, n_units, n_y_items),
        )  # [y_1, y_2, ...]
        x_seq = xp.reshape(
            xp.broadcast_to(xp.expand_dims(x_seq, axis=1), (batch, batch, n_x_items)),
            (batch * batch, n_x_items),
        )
        y_seq = xp.reshape(
            xp.broadcast_to(xp.expand_dims(y_seq, axis=0), (batch, batch, n_y_items)),
            (batch * batch, n_y_items),
        )
        xx_mask = make_attn_mask(x_seq, x_seq)
        xy_mask = make_attn_mask(y_seq, x_seq)
        yy_mask = make_attn_mask(y_seq, y_seq)
        yx_mask = make_attn_mask(x_seq, y_seq)

        for i in range(0, self.n_iterative):
            x = getattr(self, "enc_{}".format(i))(
                x, xx_mask
            )  # (batch**2, n_units, n_x_items)
            y = getattr(self, "enc_{}".format(i))(
                y, yy_mask
            )  # (batch**2, n_units, n_drops)
            x_ = x + getattr(self, "dec_{}".format(i))(
                y, x, yx_mask
            )  # (batch**2, n_units, n_x_items)
            y_ = y + getattr(self, "dec_{}".format(i))(
                x, y, xy_mask
            )  # (batch**2, n_units, n_drops)
            x = x_
            y = y_

        e = self.last_dec(x, y, yx_mask)
        score = F.reshape(e, (batch, batch))
        return score

    def embed_reshape_transpose(self, x):
        batch, n_items = x.shape[:2]
        x = F.reshape(x, (batch * n_items, x.shape[2]))
        x = self.embedder(x)  # (batch*n_items, n_units)
        x = F.reshape(x, (batch, n_items, self.n_units))
        x = F.transpose(x, (0, 2, 1))  # (batch, n_units, n_items)
        return x
