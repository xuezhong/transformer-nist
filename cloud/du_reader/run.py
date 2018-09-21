#!/usr/bin/python
"""
"""
import sys
import os
sys.path.append(r'my_Dureader/src/fluid/')

if __name__ == '__main__':
    cmd = "python   my_Dureader/src/fluid/train.py \
        --trainset my_Dureader/data/dureader/train.json \
        --devset my_Dureader/data/dureader/dev.json \
        --vocab_dir my_Dureader/data/dureader/ \
        --use_gpu true \
        --save_dir ./models \
        --pass_num 10 \
        --learning_rate 0.001 \
        --batch_size 4 \
        --embed_size 300 \
        --hidden_size 150 \
        --max_p_num 5 \
        --max_p_len 500 \
        --max_q_len 60 \
        --max_a_len 200 \
        --drop_rate 0.2 \
        --simple_net 3 $@"
    exit(os.system(cmd))
