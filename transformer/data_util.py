import os
import tarfile
import glob

import random

START_MARK = 0
END_MARK = 1
UNK_MARK = 2


class DataLoader(object):
    def __init__(self,
                 fpattern,
                 batch_size,
                 token_batch_size=0,
                 tar_fname=None,
                 sort_by_length=True,
                 shuffle=True,
                 min_len=0,
                 max_len=100):
        self._batch_size = batch_size
        self._token_batch_size = token_batch_size
        self._tar_fname = tar_fname
        self._sort_by_length = sort_by_length
        self._shuffle = shuffle
        self._min_len = min_len
        self._max_len = max_len

        src_seq_words, trg_seq_words = self._load_data(fpattern, tar_fname)
        self._src_seq_words = [[START_MARK] + src_seq + [END_MARK] for src_seq in src_seq_words]
        self._trg_seq_words = [[START_MARK] + trg_seq + [END_MARK] for trg_seq in trg_seq_words]

        self._ins_cnt = len(self._src_seq_words)
        assert len(self._trg_seq_words) == self._ins_cnt

        self._ins_idx = [i for i in xrange(self._ins_cnt)]

        if sort_by_length:
            self._sort_index_by_len()

        # fix the batch
        self._compose_batch_idx()

        self._epoch_idx = 0
        self._cur_batch_idx = 0

    def _parse_file(self, f_obj):
        src_seq_words = []
        trg_seq_words = []
        for line in f_obj:
            fields = line.strip().split('\t')
            is_valid = True
            line_words = []

            for i, field in enumerate(fields):
                words = field.split()
                if len(words) == 0 or \
                   len(words) < self._min_len or \
                   len(words) > self._max_len:
                    is_valid = False
                    break
                line_words.append(words)

            if not is_valid: continue

            assert len(line_words) == 2

            src_seq_words.append(line_words[0])
            trg_seq_words.append(line_words[1])

        return (src_seq_words, trg_seq_words)

    def _load_data(self, fpattern, tar_fname=None):
        fpaths = glob.glob(fpattern)
        src_seq_words = []
        trg_seq_words = []

        for fpath in fpaths:
            if tarfile.is_tarfile(fpath):
                assert tar_fname is not None
                f = tarfile.open(fpath, 'r')
                one_file_data = self._parse_file(f.extractfile(tar_fname))
            else:
                assert os.path.isfile(fpath)
                one_file_data = self._parse_file(open(fpath, 'r'))

            part_src_words, part_trg_words = one_file_data

            if len(src_seq_words) == 0:
                src_seq_words, trg_seq_words = part_src_words, part_trg_words
                continue

            src_seq_words.extend(part_src_words)
            trg_seq_words.extend(part_trg_words)

        return src_seq_words, trg_seq_words

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])

    def __next__(self):
        return self.next()

    def _compose_batch_idx(self):
        self._epoch_batch_idx = []

        idx = 0

        if self._token_batch_size > 0:
            batch_idx = []
            max_src_len = 0
            max_trg_len = 0
            while idx < self._ins_cnt:
                max_src_len = max(len(self._src_seq_words[self._ins_idx[idx]]),
                                  max_src_len)
                max_trg_len = max(len(self._trg_seq_words[self._ins_idx[idx]]),
                                  max_trg_len)
                max_len = max(max_src_len, max_trg_len)
                if max_len * (len(batch_idx) + 1) > self._token_batch_size:
                    self._epoch_batch_idx.append(batch_idx)
                    max_src_len = 0
                    max_trg_len = 0
                    batch_idx = []
                    continue
                batch_idx.append(self._ins_idx[idx])
                idx += 1
            if len(batch_idx) > 0:
                self._epoch_batch_idx.append(batch_idx)
        else:
            while idx < self._ins_cnt:
                batch_idx = self._ins_idx[idx:idx + self._batch_size]
                if len(batch_idx) > 0:
                    self._epoch_batch_idx.append(batch_idx)
                idx += len(batch_idx)

        if self._shuffle:
            if not self._sort_by_length and self._token_batch_size == 0:
                random.shuffle(self._ins_idx)
                self._src_seq_words = [
                    self._src_seq_words[ins_idx] for ins_idx in self._ins_idx
                ]
                self._trg_seq_words = [
                    self._trg_seq_words[ins_idx] for ins_idx in self._ins_idx
                ]
            else:
                random.shuffle(self._epoch_batch_idx)

    def _sort_index_by_len(self):
        self._ins_idx.sort(
                key=lambda idx: max(
                    len(self._src_seq_words[idx]),
                    len(self._trg_seq_words[idx])))

    def next(self):
        while self._cur_batch_idx < len(self._epoch_batch_idx):
            batch_idx = self._epoch_batch_idx[self._cur_batch_idx]
            src_seq_words = [self._src_seq_words[idx] for idx in batch_idx]
            trg_seq_words = [self._trg_seq_words[idx] for idx in batch_idx]
            # consider whether drop
            self._cur_batch_idx += 1
            return zip(src_seq_words,
                       [trg_seq[:-1] for trg_seq in trg_seq_words],
                       [trg_seq[1:] for trg_seq in trg_seq_words])

        if self._cur_batch_idx >= len(self._epoch_batch_idx):
            self._epoch_idx += 1
            self._cur_batch_idx = 0
            if self._shuffle:
                if not self._sort_by_length and self._token_batch_size == 0:
                    random.shuffle(self._ins_idx)
                    self._src_seq_words = [
                        self._src_seq_words[ins_idx]
                        for ins_idx in self._ins_idx
                    ]
                    self._trg_seq_words = [
                        self._trg_seq_words[ins_idx]
                        for ins_idx in self._ins_idx
                    ]
                else:
                    random.shuffle(self._epoch_batch_idx)
            raise StopIteration

