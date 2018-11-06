################################## User Define Configuration ###########################
################################## Data Configuration ##################################
#type of storage cluster
storage_type="hdfs"
#attention: files for training should be put on hdfs
##the list contains all file locations should be specified here
fs_name="hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310"
##If force_reuse_output_path is True ,paddle will remove output_path without check output_path exist
force_reuse_output_path="True"
##ugi of hdfs
fs_ugi=idl-dl,idl-dl@admin
#the initial model path on hdfs used to init parameters
#init_model_path=
#the initial model path for pservers
#pserver_model_dir=
#which pass
#pserver_model_pass=
#example of above 2 args:
#if set pserver_model_dir to /app/paddle/models
#and set pserver_model_pass to 123
#then rank 0 will download model from /app/paddle/models/rank-00000/pass-00123/
#and rank 1 will download model from /app/paddle/models/rank-00001/pass-00123/, etc.
##train data path on hdfs
train_data_path="/app/ecom/fcr-ad/qiuxuezhong/paddle/data/train/"
##test data path on hdfs, can be null or not setted
#test_data_path="/app/ecom/fcr-ad/qiuxuezhong/paddle/"
#the output directory on hdfs
output_path="/app/ecom/fcr-ad/qiuxuezhong/paddle/output"
thirdparty_path="/app/ecom/fcr-ad/qiuxuezhong/paddle/thirdparty"
