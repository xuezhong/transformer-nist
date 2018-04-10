import paddle.fluid as fluid
from prepare_data import create_or_get_data
from config import ModelHyperParams, TrainTaskConfig
from model import transformer_pe
import sys
import numpy


def main():
    startup = fluid.Program()
    main = fluid.Program()
    multi_files = False
    field_helper = create_or_get_data(single_file=not multi_files)
    with fluid.program_guard(main, startup):
        fileds = field_helper.create_reader(use_open_files=multi_files)

        sum_cost, avg_cost, predict, token_num = transformer_pe(
            fileds,
            ModelHyperParams.src_vocab_size + 0,
            ModelHyperParams.trg_vocab_size + 0, ModelHyperParams.max_length + 1,
            ModelHyperParams.n_layer, ModelHyperParams.n_head,
            ModelHyperParams.d_key, ModelHyperParams.d_value,
            ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
            ModelHyperParams.dropout, ModelHyperParams.src_pad_idx,
            ModelHyperParams.trg_pad_idx, ModelHyperParams.pos_pad_idx)

        optimizer = fluid.optimizer.Adam(
            learning_rate=1e-3,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps)

        optimizer.minimize(sum_cost)

        exe = fluid.Executor(fluid.CUDAPlace(0))
        exe.run(startup)

        exe = fluid.ParallelExecutor(loss_name=sum_cost.name, use_cuda=True)

        for i in xrange(sys.maxint):
            if i % 10 == 0:
                cost_np = map(numpy.array, exe.run(fetch_list=[sum_cost.name]))[0]
                print 'Batch {0}, Cost {1}'.format(i, cost_np[0])
            else:
                exe.run(fetch_list=[])


if __name__ == '__main__':
    main()
