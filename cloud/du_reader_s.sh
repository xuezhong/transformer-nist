paddlecloud job train --cluster-name paddle-jpaas-ai00-gpu \
--job-version paddle-fluid-custom \
--k8s-gpu-type baidu/gpu_p40 \
--k8s-gpu-cards 8 \
--k8s-priority high \
--k8s-memory 100Gi \
--k8s-ps-memory 20Gi \
--job-name du_reader \
--k8s-wall-time 90:00:00 \
--start-cmd "python run.py" \
--job-conf du_reader/common.py \
--files du_reader/run.py du_reader/common.py \
--image-addr "registry.baidu.com/qiuxuezhong/fluid_gpu:bidaf"
#--k8s-wall-time 10:00:00 \
#--user-sk 69594fe214b15d1f8c21cdf817de6c3b \
#--user-sk 69594fe214b15d1f8c21cdf817de6c3b \
