# transformer-nist

Run transformer on dataset nist for english and chinese.

cmd example for trainning:
psserver:
export TRAINING_ROLE=PSERVER
export TRAINERS=2
export POD_IP=127.0.0.1
export PADDLE_INIT_PORT=6174
export MKL_NUM_THREADS=1 
python -u nmt_fluid.py --local 0 --ps_host=127.0.0.1:6174 --trainer_hosts=127.0.0.1:6174,127.0.0.1:6175 --device=CPU

trainer:
export TRAINING_ROLE=TRAINER
export TRAINERS=2
export POD_IP=127.0.0.1
export PADDLE_INIT_PORT=6174
export MKL_NUM_THREADS=1 
export CUDA_VISIBLE_DEVICES=2 
python -u nmt_fluid.py --local 0 --ps_host=127.0.0.1:6174 --trainer_hosts=127.0.0.1:6174,127.0.0.1:6175 --device_id=0 --batch_size=28 --test_save=1 --exit_batch_id=1999

export TRAINING_ROLE=TRAINER
export TRAINERS=2
export POD_IP=127.0.0.1
export PADDLE_INIT_PORT=6174
export MKL_NUM_THREADS=1 
export CUDA_VISIBLE_DEVICES=3
python -u nmt_fluid.py --local 0 --ps_host=127.0.0.1:6174 --trainer_hosts=127.0.0.1:6174,127.0.0.1:6175 --device_id=0 --batch_size=28 --test_save=1 --exit_batch_id=1999
