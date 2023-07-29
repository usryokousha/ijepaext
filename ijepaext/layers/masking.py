import torch

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    :return: tensor of shape [B * len(masks), len(masks[0]), D]
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, batch_size, repeats):
    N = x.shape[0] // batch_size
    x = torch.cat([
        torch.cat([x[i*batch_size:(i+1)*batch_size] for _ in range(repeats)], dim=0)
        for i in range(N)
    ], dim=0)
    return x