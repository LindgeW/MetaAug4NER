import torch


# Time Step Dropout
def timestep_dropout(inputs, p=0.5, batch_first=True):
    '''
    :param inputs: (bz, time_step, feature_size)
    :param p: probability p mask out output nodes
    :param batch_first: default True
    :return:
    '''
    if not batch_first:
        inputs = inputs.transpose(0, 1)

    batch_size, time_step, feature_size = inputs.size()
    drop_mask = inputs.data.new_full((batch_size, feature_size), 1-p)
    drop_mask = torch.bernoulli(drop_mask).div(1 - p)
    # drop_mask = drop_mask.unsqueeze(-1).expand((-1, -1, time_step)).transpose(1, 2)
    drop_mask = drop_mask.unsqueeze(1)
    return inputs * drop_mask


# Independent Dropout
def independent_dropout(x, y, p=0.5, eps=1e-12):
    '''
    :param x: (bz, time_step, feature_size)
    :param y: (bz, time_step, feature_size)
    :param p:
    :param eps:
    :return:
    '''
    x_mask = torch.bernoulli(x.data.new_full(x.shape[:2], 1 - p))
    y_mask = torch.bernoulli(y.data.new_full(y.shape[:2], 1 - p))
    scale = 3. / (2 * x_mask + y_mask + eps)
    x_mask *= scale
    y_mask *= scale
    x = x * x_mask.unsqueeze(dim=-1)
    y = y * y_mask.unsqueeze(dim=-1)
    return x, y


def independent_dropout_bi(x, y, p=0.5, eps=1e-12):
    '''
    :param x: (bz, time_step, feature_size)
    :param y: (bz, time_step, feature_size)
    :param p:
    :param eps:
    :return:
    '''
    x_mask = torch.bernoulli(x.data.new_full(x.shape[:2], 1 - p))
    y_mask = torch.bernoulli(y.data.new_full(y.shape[:2], 1 - p))
    scale = 2. / (x_mask + y_mask + eps)
    x_mask *= scale
    y_mask *= scale
    x = x * x_mask.unsqueeze(dim=-1)
    y = y * y_mask.unsqueeze(dim=-1)
    return x, y


# 以多大的概率将一个词替换为unk，既可以训练unk也是一定的regularize
def drop_words(wd_idxs, wd_unk_idx=1, wd_dropout=0.1):
    '''
    :param wd_idxs: (bz, max_len) LongTensor 词索引
    :param wd_unk_idx UNK词索引
    :param wd_dropout: 以多大的概率将词索引置成unk_idx
    :return:
    '''
    if wd_dropout > 0:
        with torch.no_grad():
            drop_probs = wd_idxs.new_full(wd_idxs.shape, fill_value=wd_dropout, dtype=torch.float)
            # dropout_word越大，越多为1位置
            drop_mask = torch.bernoulli(drop_probs).eq(1)
            non_pad_mask = wd_idxs.ne(0)  # 非填充部分mask
            mask = non_pad_mask & drop_mask  # pad的位置不为unk
            wd_idxs = wd_idxs.masked_fill(mask, wd_unk_idx)
    return wd_idxs
