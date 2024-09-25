import torch


def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query


def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label

def split_shot_query_label(label, way, shot, query, ep_per_batch=1):
    label_shape = label.shape[1:]
    label = label.view(ep_per_batch, way, shot + query, *label_shape)
    x_shot_label, x_query_label = label.split([shot, query], dim=2)
    x_shotlabel = x_shot_label.contiguous()
    x_query_label = x_query_label.contiguous().view(ep_per_batch, way * query, *label_shape)
    return x_shot_label, x_query_label