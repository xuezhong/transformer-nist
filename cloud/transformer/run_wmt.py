#!/usr/bin/python
"""
"""
import sys
import os
sys.path.append(r'thirdparty/model/transformer_cloud/')

if __name__ == '__main__':
    cmd = "python -u thirdparty/model/transformer_cloud/train.py --src_vocab_fpath thirdparty/wmt_bpe/vocab_all.bpe.32000   --trg_vocab_fpath thirdparty/wmt_bpe/vocab_all.bpe.32000   --special_token '<s>' '<e>' '<unk>'   --train_file_pattern thirdparty/wmt_bpe/train.tok.clean.bpe.32000.en-de   --use_token_batch True   --batch_size 1600   --sort_type pool   --pool_size 200000"
    exit(os.system(cmd))
