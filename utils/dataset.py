import numpy as np
import random
import torch


class DataSet(object):
    '''
    data_path-> instance对象集 -> 生成batch -> to_index (vocab) -> padding -> to_tensor
             -> 创建vocab

    bert_path -> bert_model / bert_tokenizer (vocab)

    embed_path -> pre_embeds / pre_vocab
    '''
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

'''
class BatchWrapper(object):
    def __init__(self, dl, mixup=False, mixup_args=(4, 4)):
        super(BatchWrapper, self).__init__()
        self.dl = dl
        self.mixup = mixup
        self.mixup_args = mixup_args

    def set_batch_size(self, bs):
        self.dl.set_batch_size(bs)

    def set_mixup(self, mixup, mixup_args=None):
        self.mixup = mixup
        if mixup_args is not None:
            self.mixup_args = mixup_args

    def __iter__(self):
        for batch in self.dl:
            if self.mixup:
                batch1, batch2 = [], []
                #batch_mixup_lmbd1, batch_mixup_lmbd2 = [], []
                bids = np.random.choice(range(len(batch)), len(batch)//2, replace=False)
                for i, inst in enumerate(batch):
                    #mixup_lmbd = max(0, min(1, np.random.beta(*self.mixup_args)))
                    if i in bids:
                        batch1.append(inst)
                        #batch_mixup_lmbd1.append(mixup_lmbd)
                    else:
                        batch2.append(inst)
                        #batch_mixup_lmbd2.append(1 - mixup_lmbd)

                if len(batch) % 2 != 0:
                    batch1.append(batch2[-1])
                    #batch_mixup_lmbd1.append(1 - batch_mixup_lmbd2[-1])

                batch_mixup_lmbd1 = np.random.beta(*self.mixup_args, len(batch1))
                batch_mixup_lmbd1 = np.where(batch_mixup_lmbd1 < 0, 0., batch_mixup_lmbd1)
                batch_mixup_lmbd1 = np.where(batch_mixup_lmbd1 > 1, 1., batch_mixup_lmbd1)
                batch_mixup_lmbd2 = 1. - batch_mixup_lmbd1
                batch_mixup_lmbd1 = torch.FloatTensor(batch_mixup_lmbd1).unsqueeze(1)
                batch_mixup_lmbd2 = torch.FloatTensor(batch_mixup_lmbd2).unsqueeze(1)
                yield batch1, batch_mixup_lmbd1, \
                      batch2, batch_mixup_lmbd2
            else:
                #batch_mixup_lmbd = []
                #for i in range(len(batch)):
                #    mixup_lmbd = max(0, min(1, np.random.beta(*self.mixup_args)))
                #    batch_mixup_lmbd.append(mixup_lmbd)
                batch_mixup_lmbd = np.random.beta(*self.mixup_args, len(batch))
                #batch_mixup_lmbd = np.where(batch_mixup_lmbd > 1-batch_mixup_lmbd, batch_mixup_lmbd, 1-batch_mixup_lmbd)
                batch_mixup_lmbd = np.where(batch_mixup_lmbd < 0, 0., batch_mixup_lmbd)
                batch_mixup_lmbd = np.where(batch_mixup_lmbd > 1, 1., batch_mixup_lmbd)
                batch_mixup_lmbd = torch.FloatTensor(batch_mixup_lmbd).unsqueeze(1)
                yield batch, batch_mixup_lmbd

    def __len__(self):
        return len(self.dl)
'''



class BatchWrapper(object):
    def __init__(self, dl, mixup=False, mixup_args=(8, 8)):
        super(BatchWrapper, self).__init__()
        self.dl = dl
        self.mixup = mixup
        self.mixup_args = mixup_args

    def set_mixup(self, mixup, mixup_args=None):
        self.mixup = mixup
        if mixup_args is not None:
            self.mixup_args = mixup_args

    def __iter__(self):
        for batch in self.dl:
            if self.mixup:
                batch1, batch2 = [], []
                batch_mixup_lmbd1, batch_mixup_lmbd2 = [], []
                # bids = np.random.choice(range(len(batch)), len(batch)//2, replace=False)
                # for i, inst in enumerate(batch):
                #     if i in bids:
                #         batch1.append(inst)
                #     else:
                #         batch2.append(inst)
                #
                # if len(batch) % 2 != 0:
                #     batch1.append(batch2[-1])
                # batch_mixup_lmbd1 = np.random.beta(*self.mixup_args, len(batch1))
                # batch_mixup_lmbd1 = np.where(batch_mixup_lmbd1 < 0, 0., batch_mixup_lmbd1)
                # batch_mixup_lmbd1 = np.where(batch_mixup_lmbd1 > 1, 1., batch_mixup_lmbd1)
                # batch_mixup_lmbd2 = 1. - batch_mixup_lmbd1

                batcher = SampleWrapper(batch, np.random.beta, self.mixup_args)
                for inst1, lmbd1, inst2, lmbd2 in batcher:
                    batch1.append(inst1)
                    batch_mixup_lmbd1.append(lmbd1)
                    batch2.append(inst2)
                    batch_mixup_lmbd2.append(lmbd2)
                batch_mixup_lmbd1 = torch.FloatTensor(batch_mixup_lmbd1).unsqueeze(1)
                batch_mixup_lmbd2 = torch.FloatTensor(batch_mixup_lmbd2).unsqueeze(1)
                yield batch1, batch_mixup_lmbd1, \
                      batch2, batch_mixup_lmbd2
            else:
                # batch_mixup_lmbd = []
                # for i in range(len(batch)):
                #     mixup_lmbd = max(0, min(1, np.random.beta(*self.mixup_args)))
                #     batch_mixup_lmbd.append(mixup_lmbd)
                batch_mixup_lmbd = np.random.beta(*self.mixup_args, len(batch))
                batch_mixup_lmbd = np.where(batch_mixup_lmbd < 0, 0., batch_mixup_lmbd)
                batch_mixup_lmbd = np.where(batch_mixup_lmbd > 1, 1., batch_mixup_lmbd)
                batch_mixup_lmbd = torch.FloatTensor(batch_mixup_lmbd).unsqueeze(1)
                yield batch, batch_mixup_lmbd

    def __len__(self):
        return len(self.dl)


class SampleWrapper(object):
    """
    wrapper for each sample with mixup sampler
    """
    def __init__(self, batch, mixup, mixup_args):
        self.mixup = mixup
        self.batch = batch
        self.mixup_args = mixup_args

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, item):
        # first sample
        inst1 = self.batch[item]
        # draw a random lambda ratio from distribution
        if self.mixup is not None:
            mix_lambda = max(0, min(1, self.mixup(*self.mixup_args)))
        else:
            return inst1

        if mix_lambda >= 1 or len(self) == 1:
            return inst1, mix_lambda, \
                   inst1, 0
        # second sample
        id2 = np.random.choice(np.delete(np.arange(len(self)), item))
        inst2 = self.batch[id2]
        return inst1, mix_lambda, \
               inst2, 1 - mix_lambda



class BucketDataLoader(object):
    def __init__(self, dataset, batch_size=1, key=lambda x: len(x), shuffle=False, sort_within_batch=True, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.key = key
        self.shuffle = shuffle
        self.sort_within_batch = sort_within_batch
        self.collate_fn = collate_fn

    def set_batch_size(self, bs):
        self.batch_size = bs

    def __iter__(self):
        return pool(self.dataset, self.batch_size, self.key, shuffle=self.shuffle, sort_within_batch=self.sort_within_batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def pool(data, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         shuffle=False, sort_within_batch=False):
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    for p in batch(data, batch_size * 100, batch_size_fn):
        p_batch = batch(sorted(p, key=key), batch_size, batch_size_fn) if sort_within_batch else batch(p, batch_size, batch_size_fn)
        p_batch_list = list(p_batch)
        if shuffle:
            random.shuffle(p_batch_list)

        for b in p_batch_list:
            yield b


def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch


class DataLoader2(object):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.L = len(self)
        self.cur_i = -1

    def __iter__(self):
        self.cur_i = -1
        if self.shuffle:
            random.shuffle(self.dataset)
        return self

    def __next__(self):
        self.cur_i += 1
        if self.cur_i < self.L:
            return self.dataset[self.cur_i * self.batch_size: (self.cur_i+1) * self.batch_size]
        else:
            raise StopIteration

    def has_next(self):
        return self.cur_i < self.L - 1

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
