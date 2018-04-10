import cPickle
import numpy as np
import os

import paddle  # .v2 as paddle
import paddle.fluid as fluid

import nist_data_provider
from config import TrainTaskConfig, ModelHyperParams, encoder_input_data_names, decoder_input_data_names, \
    label_data_names
from recordio_helper import FieldHelper
import multiprocessing


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   return_pos=True,
                   return_attn_bias=True,
                   return_max_len=True):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if return_pos:
        inst_pos = np.array([[
            pos_i + 1 if w_i != pad_idx else 0 for pos_i, w_i in enumerate(inst)
        ] for inst in inst_data])

        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_input(insts, input_data_names, src_pad_idx, trg_pad_idx,
                        n_head, d_model):
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    # These shape tensors are used in reshape_op.
    src_data_shape = np.array([len(insts), src_max_len, d_model], dtype="int32")
    trg_data_shape = np.array([len(insts), trg_max_len, d_model], dtype="int32")
    src_slf_attn_pre_softmax_shape = np.array(
        [-1, src_slf_attn_bias.shape[-1]], dtype="int32")
    src_slf_attn_post_softmax_shape = np.array(
        src_slf_attn_bias.shape, dtype="int32")
    trg_slf_attn_pre_softmax_shape = np.array(
        [-1, trg_slf_attn_bias.shape[-1]], dtype="int32")
    trg_slf_attn_post_softmax_shape = np.array(
        trg_slf_attn_bias.shape, dtype="int32")
    trg_src_attn_pre_softmax_shape = np.array(
        [-1, trg_src_attn_bias.shape[-1]], dtype="int32")
    trg_src_attn_post_softmax_shape = np.array(
        trg_src_attn_bias.shape, dtype="int32")

    lbl_word = pad_batch_data([inst[2] for inst in insts], trg_pad_idx, n_head,
                              False, False, False, False)
    lbl_weight = (lbl_word != trg_pad_idx).astype("float32").reshape([-1, 1])

    input_dict = dict(
        zip(input_data_names, [
            src_word, src_pos, src_slf_attn_bias, src_data_shape,
            src_slf_attn_pre_softmax_shape, src_slf_attn_post_softmax_shape,
            trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias,
            trg_data_shape, trg_slf_attn_pre_softmax_shape,
            trg_slf_attn_post_softmax_shape, trg_src_attn_pre_softmax_shape,
            trg_src_attn_post_softmax_shape, lbl_word, lbl_weight
        ]))
    return input_dict


def create_recordio_file(item):
    filename, reader_creator, i, field_helper = item
    reader_creator = nist_data_provider.reader_creator_with_file(**reader_creator)

    train_data = paddle.batch(reader_creator, batch_size=TrainTaskConfig.batch_size)
    with fluid.recordio_writer.create_recordio_writer(filename,
                                                      max_num_records=100) as writer:
        for j, batch in enumerate(train_data()):
            if len(batch) != TrainTaskConfig.batch_size:
                continue
            data_input = prepare_batch_input(
                batch, encoder_input_data_names + decoder_input_data_names[:-1] +
                       label_data_names, ModelHyperParams.src_pad_idx,
                ModelHyperParams.trg_pad_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)

            for input_name in encoder_input_data_names + decoder_input_data_names[:-1] + label_data_names:
                if input_name not in data_input:
                    continue
                tensor = data_input[input_name]
                t = fluid.LoDTensor()
                t.set(tensor, fluid.CPUPlace())
                if i == 0 and j == 0:
                    field_helper.append_field(input_name, tensor.shape, tensor.dtype)
                writer.append_tensor(t)
            writer.complete_append_tensor()
    return field_helper


def create_or_get_data(process_num=10, single_file=False):
    creators = nist_data_provider.train_creators("data", ModelHyperParams.src_vocab_size,
                                                 ModelHyperParams.trg_vocab_size)

    if single_file:
        creators = creators[:1]  # drop other files. Make test faster

    recordio_files = ["./nist06_batchsize_{0}.part{1}.recordio".format(TrainTaskConfig.batch_size, i) for i in
                      xrange(len(creators))]
    field_helpers_fn = './nist06_batchsize_{0}.recordio.fields'.format(TrainTaskConfig.batch_size)
    any_file_not_exist = reduce(lambda acc, path: acc or not os.path.exists(path), [field_helpers_fn] + recordio_files,
                                False)
    if any_file_not_exist:
        pool = multiprocessing.Pool(process_num)
        field_helper = FieldHelper(recordio_files)

        items = []
        for i, pair in enumerate(zip(recordio_files, creators)):
            items.append((pair[0], pair[1], i, field_helper))

        field_helper = pool.map(create_recordio_file, items)[0]

        with open(field_helpers_fn, 'w') as f:
            cPickle.dump(field_helper, f, cPickle.HIGHEST_PROTOCOL)
        return field_helper
    else:
        with open(field_helpers_fn, 'r') as f:
            return cPickle.load(f)


if __name__ == "__main__":
    create_or_get_data()
