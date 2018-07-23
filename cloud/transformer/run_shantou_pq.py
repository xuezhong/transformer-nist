#!/usr/bin/python
"""
"""
import sys
import os
sys.path.append(r'thirdparty/model/transformer_cloud/')

if __name__ == '__main__':
    cmd = "python -u thirdparty/model/transformer_cloud/train_simple.py --src_vocab_fpath thirdparty/dataset/shantou_data_pq/vocab --trg_vocab_fpath thirdparty/dataset/shantou_data_pq/vocab --train_file_pattern 'thirdparty/dataset/shantou_data_qp/20180426_dsa' --batch_size 256 --use_token_batch True  --shuffle False --sync True --shuffle_batch False --special_token '_GO' '_EOS' '_UNK' weight_sharing True"
    exit(os.system(cmd))
