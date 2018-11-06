#/bin/bash
batch_size=$1
optim=$2
dev_interval=$3
wd=$4
wd_v=0.0
if [ $wd = "wd" ];then
    wd_v=0.0001
fi
paddlecloud job train --cluster-name paddle-jpaas-ai00-gpu \
--job-version paddle-fluid-custom \
--k8s-gpu-type baidu/gpu_p40 \
--k8s-gpu-cards 1 \
--k8s-priority high \
--k8s-memory 100Gi \
--k8s-ps-memory 20Gi \
--job-name dr_nomp_b${batch_size}_newdata_${optim}_dev${dev_interval}_${wd} \
--k8s-wall-time 90:00:00 \
--start-cmd "env FLAGS_fraction_of_gpu_memory_to_use=0.1 python   thirdparty/src/fluid_new_dataset/train.py \
        --trainset thirdparty/data/dureader/train.json \
        --devset thirdparty/data/dureader/dev.json \
        --vocab_dir thirdparty/data/dureader/ \
        --use_gpu true \
        --save_dir ./models \
        --pass_num 10 \
        --learning_rate 0.001 \
        --batch_size ${batch_size} \
        --embed_size 300 \
        --hidden_size 150 \
        --max_p_num 5 \
        --max_p_len 500 \
        --max_q_len 60 \
        --max_a_len 200 \
        --drop_rate 0.2 \
        --optim ${optim} \
        --dev_interval ${dev_interval} \
        --weight_decay ${wd_v} \
        --simple_net 3" \
--job-conf du_reader/common.py \
--files du_reader/run.py du_reader/common.py \
--image-addr "registry.baidu.com/qiuxuezhong/fluid_gpu:2f5a7cc4700a37fdda5ed6312e9cae9646078f85"
#--k8s-wall-time 10:00:00 \
#--user-sk 69594fe214b15d1f8c21cdf817de6c3b \
#--user-sk 69594fe214b15d1f8c21cdf817de6c3b \
