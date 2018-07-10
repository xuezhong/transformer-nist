paddlecloud job train --cluster-name paddle-jpaas-ai00 \
--job-version custom-fluid \
--k8s-gpu-type baidu/gpu_p40 \
--k8s-gpu-cards 8 \
--k8s-priority high \
--k8s-memory 100Gi \
--k8s-ps-memory 20Gi \
--job-name s-transformer-wmt \
--k8s-wall-time 90:00:00 \
--start-cmd "python run_wmt.py" \
--job-conf transformer/common.py \
--files transformer/run_wmt.py transformer/common.py \
--image-addr "registry.baidu.com/qiuxuezhong/fluid_gpu:0.14.0"
#--k8s-wall-time 10:00:00 \
#--user-sk 69594fe214b15d1f8c21cdf817de6c3b \
#--user-sk 69594fe214b15d1f8c21cdf817de6c3b \
