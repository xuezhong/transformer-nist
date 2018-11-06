batch_size=$1
optim=$2
dev_interval=$3
paddlecloud job train --cluster-name paddle-jpaas-ai00-gpu \
--job-version paddle-fluid-custom \
--k8s-gpu-type baidu/gpu_p40 \
--k8s-gpu-cards 1 \
--k8s-priority high \
--k8s-memory 100Gi \
--k8s-ps-memory 20Gi \
--job-name dr_open_b${batch_size}_${optim}_dev${dev_interval}_newd \
--k8s-wall-time 5000:00:00 \
--start-cmd "python   thirdparty/src/fluid_open_source/run.py \
        --trainset thirdparty/data/old_dataset/preprocessed/trainset/*.train.json \
        --devset thirdparty/data/old_dataset/preprocessed/devset/*.dev.json \
        --vocab_dir thirdparty/data/old_dataset/vocab \
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
        --train \
        --weight_decay 0.0001 \
        --dev_interval ${dev_interval}" \
--job-conf du_reader/common.py \
--files du_reader/run.py du_reader/common.py du_reader/fs_ugi.py \
--image-addr "registry.baidu.com/qiuxuezhong/fluid_gpu:db460e8d9ed7dec26e295f1b7e9e9872314c62bf"
#--k8s-wall-time 10:00:00 \
#--user-sk 69594fe214b15d1f8c21cdf817de6c3b \
#--user-sk 69594fe214b15d1f8c21cdf817de6c3b \
