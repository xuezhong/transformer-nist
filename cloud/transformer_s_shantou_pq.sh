paddlecloud job train --cluster-name paddle-jpaas-ai00 \
--job-version custom-fluid \
--k8s-gpu-type baidu/gpu_p40 \
--k8s-gpu-cards 8 \
--k8s-wall-time 1000:00:00 \
--k8s-priority high \
--k8s-memory 200Gi \
--k8s-ps-memory 20Gi \
--job-name s-transformer-shantou \
--start-cmd "python run_shantou_pq.py" \
--job-conf transformer/common.py \
--files transformer/run_shantou_pq.py transformer/common.py \
--image-addr "registry.baidu.com/qiuxuezhong/fluid_gpu:0.14.0"
