
import os

import math
import paddle
import paddle.fluid as fluid
import sys
import ctrdata

from paddle.fluid.metrics import Auc

import argparse

def parse_args():
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default='data',
        help="The path of vocabulary file of source language.")
    return parser.parse_args()


def get_file_list_static():
    FILELIST = ['./test_shitu']
    #FILELIST = ['./0000/part-00000-S.gz']
    print("FILELIST:" + str(FILELIST))
    return FILELIST

def get_file_list():
    args = parse_args()
    data_dir = args.data_dir
    data_files = os.listdir(data_dir)
    FILELIST = list()
    for data_file in data_files:
        FILELIST.append(data_dir + '/' + data_file)
    
    print("FILELIST:" + str(FILELIST))
    return FILELIST

PASS_NUM = 10
EMBED_SIZE = 1
BATCH_SIZE = 1
IS_SPARSE = True
CNN_DIM = 128
CNN_FILTER_SIZE = 5
is_distributed = True

DICT_SIZE = 10000 * 10


def word_emb(word):
    """
    """
    embed = fluid.layers.embedding(
        input=word,
        size=[DICT_SIZE, EMBED_SIZE],
        dtype='float32',
        param_attr='shared_w',
        is_sparse=IS_SPARSE,
        is_distributed=False)
    return embed


def text_cnn(word):
    """
    """
    embed = fluid.layers.embedding(
        input=word,
        size=[DICT_SIZE, EMBED_SIZE],
        dtype='float32',
        is_sparse=IS_SPARSE,
        is_distributed=False)
    cnn = fluid.nets.sequence_conv_pool(
         input = embed,
         num_filters = CNN_DIM,
         filter_size = CNN_FILTER_SIZE,
         pool_type = "max")
    return cnn



def train(pserver_endpoints,
          training_role,
          current_endpoint,
          trainer_num,
          trainer_id,
          use_cuda=False, is_sparse=True, is_local=True):

    age = fluid.layers.data(name='firstw0', shape=[1], dtype='int64')
    gender = fluid.layers.data(name='firstw1', shape=[1], dtype='int64')
    preq_ids = fluid.layers.data(name='firstw2', shape=[1], dtype='int64', lod_level=1)
    query_profile_ids = fluid.layers.data(name='firstw3', shape=[1], dtype='int64', lod_level=1)
    click_word_profile_ids = fluid.layers.data(name='firstw4', shape=[1], dtype='int64', lod_level=1)
    click_title_profile_ids = fluid.layers.data(name='firstw5', shape=[1], dtype='int64', lod_level=1)
    total_profile_fea = fluid.layers.data(name='firstw6', shape=[2], dtype='float32')
    cmatch_profile_fea = fluid.layers.data(name='firstw7', shape=[4], dtype='float32')
    mtid_profile_fea = fluid.layers.data(name='firstw8', shape=[4], dtype='float32')
    cmatch = fluid.layers.data(name='firstw9', shape=[1], dtype='int64')
    ua = fluid.layers.data(name='firstw10', shape=[1], dtype='int64', lod_level=1)
    mtid = fluid.layers.data(name='firstw11', shape=[1], dtype='int64')
    title_ids = fluid.layers.data(name='firstw12', shape=[1], dtype='int64', lod_level=1)
    brand_ids = fluid.layers.data(name='firstw13', shape=[1], dtype='int64', lod_level=1)
    word_ids = fluid.layers.data(name='firstw14', shape=[1], dtype='int64', lod_level=1)

   
    vecs = [word_emb(age), \
            word_emb(gender), \
            text_cnn(preq_ids), \
            text_cnn(query_profile_ids), \
            text_cnn(click_title_profile_ids), \
            text_cnn(click_word_profile_ids), \
            total_profile_fea, \
            cmatch_profile_fea, \
            mtid_profile_fea, \
            word_emb(cmatch), \
            text_cnn(ua), \
            word_emb(mtid), \
            text_cnn(title_ids), \
            text_cnn(brand_ids), \
            text_cnn(word_ids)
           ] 

    fc0 = fluid.layers.concat(input=vecs, axis = 1)

    fc1 = fluid.layers.fc(input=fc0, size=512, act="relu")
    fc2 = fluid.layers.fc(input=fc1, size=128, act="relu")
    fc3 = fluid.layers.fc(input=fc2, size=128, act="relu")
    predict = fluid.layers.fc(input=fc3, size=2, act="softmax")

    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.003)
    optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    data_list = [age, 
        gender,
        preq_ids,
        query_profile_ids,
        click_word_profile_ids,
        click_title_profile_ids,
        total_profile_fea,
        cmatch_profile_fea,
        mtid_profile_fea,
        cmatch,
        ua,
        mtid,
        title_ids,
        brand_ids,
        word_ids, 
        label]

    feeder = fluid.DataFeeder(feed_list=data_list, place=place)
    
    ctr = ctrdata.CTRData()

    def train_loop(main_program):
        # prepare data
        train_data = paddle.batch(
            paddle.reader.shuffle(
                ctr.train(get_file_list()), buf_size=1024 * 100),
            batch_size=BATCH_SIZE)

        exe.run(fluid.default_startup_program())

        print("start to run")
        for pass_id in xrange(PASS_NUM):
            auc = Auc(name="auc")
            batch_id = 0
            for data in train_data():
                print(data)
                cost_val, acc_val, label_val, predict_val = exe.run(main_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_cost, accuracy, label, predict])
                label_val = label_val.reshape(len(label_val))
                auc.update(predict_val, label_val)
                if batch_id % 10 == 0:
                    #print("label=" + str(label_val))
                    #print("predict=" + str(predict_val))
                    print("pass_id=" + str(pass_id) + " batch_id=" + str(batch_id) + " cost=" + str(cost_val) + " acc=" + str(acc_val))
                    if math.isnan(float(cost_val)):
                        sys.exit("got NaN loss, training failed.")
                batch_id += 1
            print("pass_id=" + str(pass_id) + " auc=" + str(auc.eval()))

    if is_local:
        train_loop(fluid.default_main_program())
    else:
        print("dist train")
        t = fluid.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainer_num)
        if training_role == "PSERVER":
            print("run pserver")
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            print("run trianer")
            train_loop(t.get_trainer_program())

if __name__ == '__main__':
    port = os.getenv("PADDLE_PORT", "6174")
    pserver_ips = os.getenv("PADDLE_PSERVERS", "")  # ip,ip...
    eplist = []
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
    trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    current_endpoint = os.getenv("POD_IP", "localhost") + ":" + port
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    training_role = os.getenv("TRAINING_ROLE", "TRAINER")
    is_local = bool(int(os.getenv("PADDLE_IS_LOCAL", 1)))

    train(
        is_local=is_local,
        training_role=training_role,
        pserver_endpoints=pserver_endpoints, # ip:port,ip:port list
        current_endpoint=current_endpoint, # ip:port the ip port for this pserver
        trainer_num=trainer_num,
        trainer_id=trainer_id,
    )
