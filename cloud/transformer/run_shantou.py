#!/usr/bin/python
"""
"""
import sys
import os
sys.path.append(r'thirdparty/model/transformer_cloud/')

if __name__ == '__main__':
    cmd = "python -u thirdparty/model/transformer_cloud/train_simple.py --src_vocab_fpath thirdparty/dataset/shantou_data/vocab --trg_vocab_fpath thirdparty/dataset/shantou_data/vocab --train_file_pattern 'thirdparty/dataset/shantou_data/20180426' --batch_size 128 --use_token_batch True  --shuffle False --sync False --shuffle_batch False --special_token '_GO' '_EOS' '_UNK' weight_sharing True"
    exit(os.system(cmd))
