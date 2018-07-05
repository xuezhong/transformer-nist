import os
import tarfile
import glob

import random


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"


class EndEpoch():
    pass


class Pool(object):
    def __init__(self, sample_generator, pool_size, sort):
        self._pool_size = pool_size
        self._pool = []
        self._sample_generator = sample_generator()
        self._end = False
        self._sort = sort

    def _fill(self):
        while len(self._pool) < self._pool_size and not self._end:
            try:
                sample = self._sample_generator.next()
                self._pool.append(sample)
            except StopIteration as e:
                self._end = True
                break

        if self._sort:
            self._pool.sort(
                key=lambda sample: max(len(sample[0]), len(sample[1])) \
                if len(sample) > 1 else len(sample[0])
            )

        if self._end and len(self._pool) < self._pool_size:
            self._pool.append(EndEpoch())

    def push_back(self, samples):
        if len(self._pool) != 0:
            raise Exception("Pool should be empty.")

        if len(samples) >= self._pool_size:
            raise Exception("Capacity of pool should be greater than a batch. "
                            "Please enlarge `pool_size`.")

        for sample in samples:
            self._pool.append(sample)

        self._fill()

    def next(self, look=False):
        if len(self._pool) == 0:
            return None
        else:
            return self._pool[0] if look else self._pool.pop(0)


class DataReader(object):
    """
    The data reader loads all data from files and produces batches of data
    in the way corresponding to settings.

    An example of returning a generator producing data batches whose data
    is shuffled in each pass and sorted in each pool:

    ```
    train_data = DataReader(
        src_vocab_fpath='data/src_vocab_file',
        trg_vocab_fpath='data/trg_vocab_file',
        fpattern='data/part-*',
        use_token_batch=True,
        batch_size=2000,
        pool_size=10000,
        sort_type=SortType.POOL,
        shuffle=True,
        shuffle_batch=True,
        start_mark='<s>',
        end_mark='<e>',
        unk_mark='<unk>',
        clip_last_batch=False).batch_generator
    ```

    :param src_vocab_fpath: The path of vocabulary file of source language.
    :type src_vocab_fpath: basestring
    :param trg_vocab_fpath: The path of vocabulary file of target language.
    :type trg_vocab_fpath: basestring
    :param fpattern: The pattern to match data files.
    :type fpattern: basestring
    :param batch_size: The number of sequences contained in a mini-batch.
        or the maximum number of tokens (include paddings) contained in a
        mini-batch.
    :type batch_size: int
    :param pool_size: The size of pool buffer.
    :type pool_size: int
    :param sort_type: The grain to sort by length: 'global' for all
        instances; 'pool' for instances in pool; 'none' for no sort.
    :type sort_type: basestring
    :param clip_last_batch: Whether to clip the last uncompleted batch.
    :type clip_last_batch: bool
    :param tar_fname: The data file in tar if fpattern matches a tar file.
    :type tar_fname: basestring
    :param min_length: The minimum length used to filt sequences.
    :type min_length: int
    :param max_length: The maximum length used to filt sequences.
    :type max_length: int
    :param shuffle: Whether to shuffle all instances.
    :type shuffle: bool
    :param shuffle_batch: Whether to shuffle the generated batches.
    :type shuffle_batch: bool
    :param use_token_batch: Whether to produce batch data according to
        token number.
    :type use_token_batch: bool
    :param delimiter: The delimiter used to split source and target in each
        line of data file.
    :type delimiter: basestring
    :param start_mark: The token representing for the beginning of
        sentences in dictionary.
    :type start_mark: basestring
    :param end_mark: The token representing for the end of sentences
        in dictionary.
    :type end_mark: basestring
    :param unk_mark: The token representing for unknown word in dictionary.
    :type unk_mark: basestring
    :param seed: The seed for random.
    :type seed: int
    """

    def __init__(self,
                 src_vocab_fpath,
                 trg_vocab_fpath,
                 fpattern,
                 batch_size,
                 pool_size,
                 sort_type=SortType.NONE,
                 clip_last_batch=True,
                 tar_fname=None,
                 min_length=0,
                 max_length=100,
                 shuffle=True,
                 shuffle_batch=False,
                 use_token_batch=False,
                 delimiter="\t",
                 start_mark="<s>",
                 end_mark="<e>",
                 unk_mark="<unk>",
                 seed=0):
        self._fpattern = fpattern
        self._src_vocab = self.load_dict(src_vocab_fpath)
        self._only_src = True
        if trg_vocab_fpath is not None:
            self._trg_vocab = self.load_dict(trg_vocab_fpath)
            self._only_src = False
        self._pool_size = pool_size
        self._batch_size = batch_size
        self._use_token_batch = use_token_batch
        self._sort_type = sort_type
        self._clip_last_batch = clip_last_batch
        self._shuffle = shuffle
        self._shuffle_batch = shuffle_batch
        self._min_length = min_length
        self._max_length = max_length
        self._delimiter = delimiter
        self._epoch_batches = []
        self._start_mark = start_mark
        self._end_mark = end_mark
        self._unk_mark = unk_mark
        random.seed(seed)

    def _parse_file(self, fpath):
        f_obj = open(fpath, 'r')
        for line in f_obj:
            fields = line.strip().split(self._delimiter)

            if len(fields) != 2 or (self._only_src and len(fields) != 1):
                continue

            sample_words = []
            is_valid_sample = True
            max_len = -1

            for i, seq in enumerate(fields):
                seq_words = seq.split()
                max_len = max(max_len, len(seq_words))
                if len(seq_words) == 0 or \
                        len(seq_words) < self._min_length or \
                        len(seq_words) > self._max_length or \
                        (self._use_token_batch and max_len > self._batch_size):
                    is_valid_sample = False
                    break

                sample_words.append(seq_words)

            if not is_valid_sample: continue

            #print sample_words[0], sample_words[1]
            yield sample_words[0], sample_words[1]

    @staticmethod
    def load_dict(dict_path, reverse=False):
        word_dict = {}
        with open(dict_path, "r") as fdict:
            for idx, line in enumerate(fdict):
                if reverse:
                    word_dict[idx] = line.strip()
                else:
                    word_dict[line.strip()] = idx
        return word_dict

    def _sample_generator(self):
        fpaths = glob.glob(self._fpattern)

        for fpath in fpaths:
            if not os.path.isfile(fpath):
                raise IOError("Invalid file: %s" % fpath)
            
            line = self._parse_file(fpath) 
            while(True):
                try:
                    src_seq_word, trg_seq_word = line.next()

                    #print src_seq_word, trg_seq_word
                    src_seq_id = [
                        self._src_vocab.get(word, self._src_vocab.get(self._unk_mark))
                        for word in ([self._start_mark] + src_seq_word + [self._end_mark])
                    ]
                    trg_seq_id = [
                        self._src_vocab.get(word, self._src_vocab.get(self._unk_mark))
                        for word in ([self._start_mark] + trg_seq_word + [self._end_mark])
                    ]
                    yield (src_seq_id, trg_seq_id[:-1], trg_seq_id[1:])
                
                except StopIteration as e:
                    break


    def batch_generator(self):
        pool = Pool(self._sample_generator, self._pool_size, True
                    if self._sort_type == SortType.POOL else False)

        def next_batch():
            batch_data = []
            max_len = -1
            batch_max_seq_len = -1

            while True:
                sample = pool.next(look=True)

                if sample is None:
                    pool.push_back(batch_data)
                    batch_data = []
                    continue

                if isinstance(sample, EndEpoch):
                    return batch_data, batch_max_seq_len, True

                max_len = max(max_len, len(sample[0]))

                if not self._only_src:
                    max_len = max(max_len, len(sample[1]))

                if self._use_token_batch:
                    if max_len * (len(batch_data) + 1) < self._batch_size:
                        batch_max_seq_len = max_len
                        batch_data.append(pool.next())
                    else:
                        return batch_data, batch_max_seq_len, False
                else:
                    if len(batch_data) < self._batch_size:
                        batch_max_seq_len = max_len
                        batch_data.append(pool.next())
                    else:
                        return batch_data, batch_max_seq_len, False

        if not self._shuffle_batch:
            batch_data, batch_max_seq_len, last_batch = next_batch()
            while not last_batch:
                yield batch_data
                batch_data, batch_max_seq_len, last_batch = next_batch()

            batch_size = len(batch_data)
            if self._use_token_batch:
                batch_size *= batch_max_seq_len

            if (not self._clip_last_batch and len(batch_data) > 0) \
                    or (batch_size == self._batch_size):
                yield batch_data
        else:
            # should re-generate batches
            if self._sort_type == SortType.POOL \
                    or len(self._epoch_batches) == 0:
                self._epoch_batches = []
                batch_data, batch_max_seq_len, last_batch = next_batch()
                while not last_batch:
                    self._epoch_batches.append(batch_data)
                    batch_data, batch_max_seq_len, last_batch = next_batch()

                batch_size = len(batch_data)
                if self._use_token_batch:
                    batch_size *= batch_max_seq_len

                if (not self._clip_last_batch and len(batch_data) > 0) \
                        or (batch_size == self._batch_size):
                    self._epoch_batches.append(batch_data)

            random.shuffle(self._epoch_batches)

            for batch_data in self._epoch_batches:
                yield batch_data
