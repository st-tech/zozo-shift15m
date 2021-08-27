def make_attn_mask(source, target):
    mask = (target[:, None, :] >= 0) * (source[:, :, None] >= 0)
    # (batch, source_length, target_length)
    return mask
