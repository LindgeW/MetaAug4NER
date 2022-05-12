import numpy as np
import random
import torch
from torch.distributions import Beta


class DataSet(object):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class MyDataSet(DataSet):
    def __init__(self, insts, transform=None):
        self.insts = insts
        self.transform = transform

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        sample = self.insts[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __iter__(self):
        for inst in self.insts:
            yield inst

    def index(self, item):
        return self.insts.index(item)

    def data_split(self, split_rate=0.33, shuffle=False):
        assert self.insts and len(self.insts) > 0
        if shuffle:
            np.random.shuffle(self.insts)
        val_size = int(len(self.insts) * split_rate)
        train_set = MyDataSet(self.insts[:-val_size])
        val_set = MyDataSet(self.insts[-val_size:])
        return train_set, val_set


def data_split(data_set, split_rate: list, shuffle=False):
    assert len(data_set) != 0, 'Empty dataset !'
    assert len(split_rate) != 0, 'Empty split rate list !'

    n = len(data_set)
    if shuffle:
        range_idxs = np.random.permutation(n)
    else:
        range_idxs = np.asarray(range(n))

    k = 0
    parts = []
    base = sum(split_rate)
    for i, part in enumerate(split_rate):
        part_size = int((part / base) * n)
        parts.append([data_set[j] for j in range_idxs[k: k+part_size]])
        k += part_size
    return tuple(parts)


class DataLoader(object):
    def __init__(self, dataset: list, batch_size=1, shuffle=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            idxs = np.random.permutation(n)
        else:
            idxs = range(n)
        batch = []
        for idx in idxs:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                if self.collate_fn:
                    yield self.collate_fn(batch)
                else:
                    yield batch
                batch = []

        if len(batch) > 0:
            if self.collate_fn:
                yield self.collate_fn(batch)
            else:
                yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DataLoader2(object):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.len = len(self)
        self.cur_i = -1
        self.ids = None

    def __iter__(self):
        self.cur_i = -1
        self.ids = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.ids)
        return self

    def __next__(self):
        self.cur_i += 1
        if self.cur_i < self.len:
            batch_ids = self.ids[self.cur_i*self.batch_size: (1+self.cur_i)*self.batch_size]
            return [self.dataset[i] for i in batch_ids]
        else:
            raise StopIteration

    def has_next(self):
        return self.cur_i < self.len - 1

    def __len__(self):
        # return len(self.dataset) // self.batch_size    # drop last
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class BatchWrapper(object):
    def __init__(self, dl, mixup=False, mixup_args=(7, 7)):
        super(BatchWrapper, self).__init__()
        self.dl = dl
        self.mixup = mixup
        self.mixup_args = mixup_args
        self.beta_dist = Beta(*mixup_args)

    def set_mixup(self, mixup, mixup_args=None):
        self.mixup = mixup
        if mixup_args is not None:
            self.mixup_args = mixup_args

    def __iter__(self):
        for batch in self.dl:
            if self.mixup:
                batch1, batch2 = [], []
                bids = np.random.choice(range(len(batch)), len(batch)//2, replace=False)
                for i, inst in enumerate(batch):
                    if i in bids:
                        batch1.append(inst)
                    else:
                        batch2.append(inst)
                if len(batch) % 2 != 0:
                    batch1.append(batch2[0])
                assert len(batch1) == len(batch2)
                # batch_mixup_alpha = torch.tensor(np.random.beta(*self.mixup_args, (len(batch1), 1))).float()
                batch_mixup_alpha = self.beta_dist.sample((len(batch1), 1))
                yield batch1, batch_mixup_alpha, batch2, 1-batch_mixup_alpha
            else:
                # batch_mixup_alpha = torch.tensor(np.random.beta(*self.mixup_args, (len(batch), 1))).float()
                batch_mixup_alpha = self.beta_dist.sample((len(batch), 1))
                yield batch, batch_mixup_alpha

    def __len__(self):
        return len(self.dl)


# class BatchWrapper(object):
#     def __init__(self, dl, mixup=False, mixup_args=(7, 7)):
#         super(BatchWrapper, self).__init__()
#         self.dl = dl
#         self.mixup = mixup
#         self.mixup_args = mixup_args
#         self.beta_dist = Beta(*mixup_args)
#
#     def set_mixup(self, mixup, mixup_args=None):
#         self.mixup = mixup
#         if mixup_args is not None:
#             self.mixup_args = mixup_args
#
#     def __iter__(self):
#         for batch in self.dl:
#             if self.mixup:
#                 rand_ids = np.random.permutation(len(batch))
#                 batch2 = [batch[i] for i in rand_ids]
#                 assert len(batch) == len(batch2)
#                 # batch_mixup_alpha = torch.tensor(np.random.beta(*self.mixup_args, (len(batch1), 1))).float()
#                 batch_mixup_alpha = self.beta_dist.sample((len(batch), 1))
#                 yield batch, batch_mixup_alpha, batch2, 1-batch_mixup_alpha
#             else:
#                 # batch_mixup_alpha = torch.tensor(np.random.beta(*self.mixup_args, (len(batch), 1))).float()
#                 batch_mixup_alpha = self.beta_dist.sample((len(batch), 1))
#                 yield batch, batch_mixup_alpha
#
#     def __len__(self):
#         return len(self.dl)


class BucketDataLoader(object):
    def __init__(self, dataset, batch_size=1, key=lambda x: len(x), shuffle=False, sort_within_batch=True, train=True, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.key = key
        self.shuffle = shuffle
        self.train = train
        self.sort_within_batch = sort_within_batch
        self.collate_fn = collate_fn

    def set_batch_size(self, bs):
        self.batch_size = bs

    def __iter__(self):
        if not self.train:
            return batch(self.dataset, self.batch_size)
        else:
            return pool(self.dataset, self.batch_size, self.key, shuffle=self.shuffle, sort_within_batch=self.sort_within_batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def pool(data, batch_size, key, shuffle=False, sort_within_batch=False):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    for p in batch(data, batch_size * 100):
        p_batch = batch(sorted(p, key=key), batch_size) if sort_within_batch else batch(p, batch_size)
        p_batch_list = list(p_batch)
        if shuffle:
            random.shuffle(p_batch_list)
        for b in p_batch_list:
            yield b


def batch(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = len(minibatch)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], 1
    if minibatch:
        yield minibatch


