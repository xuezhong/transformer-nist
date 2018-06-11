paddlecloud job train --cluster_name paddle-jpaas-ai00 \
--job_version fluid \
--k8s_gpu_type baidu/gpu_p40 \
--k8s_gpu_cards 4 \
--k8s_priority high \
--k8s_walltime 10:00:00 \
--k8s_memory 20Gi \
s_nmt nmt/run.py nmt/common.py
