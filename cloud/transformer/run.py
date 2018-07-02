#!/usr/bin/python
"""
"""
import sys
import os
sys.path.append(r'thirdparty/model/transformer_cloud/')

if __name__ == '__main__':
    cmd = "python -u thirdparty/model/transformer_cloud/train.py --src_vocab_fpath ./thirdparty/nist06n/cn_30001.dict --trg_vocab_fpath ./thirdparty/nist06n/en_30001.dict --train_file_pattern './train/part-*' --batch_size 1024 --use_token_batch True  --sync False --special_token '_GO' '_EOS' '_UNK'"
    exit(os.system(cmd))
