class TrainTaskConfig(object):
    use_gpu = True
    # the epoch number to train.
    pass_num = 100

    # the number of sequences contained in a mini-batch.
    batch_size = 64  # there are memleak in Op::DataTransform. Need double buffer

    # the hyper parameters for Adam optimizer.
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.98
    eps = 1e-9

    # the parameters for learning rate scheduling.
    warmup_steps = 4000

    # the flag indicating to use average loss or sum loss when training.
    use_avg_cost = False

    # the directory for saving trained models.
    model_dir = "trained_models"

    #set default 
    token_batch_size = 0
    sort_by_length = False


class InferTaskConfig(object):
    use_gpu = True
    # the number of examples in one run for sequence generation.
    batch_size = 10

    # the parameters for beam search.
    beam_size = 5
    max_length = 100
    # the number of decoded sentences to output.
    n_best = 1

    # the flags indicating whether to output the special tokens.
    output_bos = False
    output_eos = False
    output_unk = False

    # the directory for loading the trained model.
    model_path = "../../pass_29.infer.model"


class ModelHyperParams(object):
    # Dictionary size for source and target language. This model directly uses
    # paddle.dataset.wmt16 in which <bos>, <eos> and <unk> token has
    # alreay been added, but the <pad> token is not added. Transformer requires
    # sequences in a mini-batch are padded to have the same length. A <pad> token is
    # added into the original dictionary in paddle.dateset.wmt16.

    # size of source word dictionary.
    src_vocab_size = 30002
    # index for <pad> token in source language.
    src_pad_idx = 0

    # size of target word dictionay
    trg_vocab_size = 30002
    # index for <pad> token in target language.
    trg_pad_idx = 0

    # index for <bos> token
    bos_idx = 1
    # index for <eos> token
    eos_idx = 2
    # index for <unk> token
    unk_idx = 3

    # position value corresponding to the <pad> token.
    pos_pad_idx = 0

    # max length of sequences. It should plus 1 to include position
    # padding token for position encoding.
    max_length = 150

    # the dimension for word embeddings, which is also the last dimension of
    # the input and output of multi-head attention, position-wise feed-forward
    # networks, encoder and decoder.

    d_model = 512
    # size of the hidden layer in position-wise feed-forward networks.
    d_inner_hid = 2048
    # the dimension that keys are projected to for dot-product attention.
    d_key = 64
    # the dimension that values are projected to for dot-product attention.
    d_value = 64
    # number of head used in multi-head attention.
    n_head = 8
    # number of sub-layers to be stacked in the encoder and decoder.
    n_layer = 6
    # dropout rate used by all dropout layers.
    dropout = 0.


# Names of position encoding table which will be initialized externally.
pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table", )

# Names of all data layers in encoder listed in order.
encoder_input_data_names = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias",
    "src_data_shape",
    "src_slf_attn_pre_softmax_shape",
    "src_slf_attn_post_softmax_shape", )

# Names of all data layers in decoder listed in order.
decoder_input_data_names = (
    "trg_word",
    "trg_pos",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "trg_data_shape",
    "trg_slf_attn_pre_softmax_shape",
    "trg_slf_attn_post_softmax_shape",
    "trg_src_attn_pre_softmax_shape",
    "trg_src_attn_post_softmax_shape",
    "enc_output", )

# Names of label related data layers listed in order.
label_data_names = (
    "lbl_word",
    "lbl_weight", )

data_shapes = {
    "src_word": (1 * (ModelHyperParams.max_length + 1), 1L),
    "src_pos": (1 * (ModelHyperParams.max_length + 1), 1L),
    "src_slf_attn_bias":
    (1, ModelHyperParams.n_head, (ModelHyperParams.max_length + 1),
     (ModelHyperParams.max_length + 1)),
    "src_data_shape": (3L, ),
    "src_slf_attn_pre_softmax_shape": (2L, ),
    "src_slf_attn_post_softmax_shape": (4L, ),
    "trg_word": (1 * (ModelHyperParams.max_length + 1), 1L),
    "trg_pos": (1 * (ModelHyperParams.max_length + 1), 1L),
    "trg_slf_attn_bias": (1, ModelHyperParams.n_head,
                          (ModelHyperParams.max_length + 1),
                          (ModelHyperParams.max_length + 1)),
    "trg_src_attn_bias": (1, ModelHyperParams.n_head,
                          (ModelHyperParams.max_length + 1),
                          (ModelHyperParams.max_length + 1)),
    "trg_data_shape": (3L, ),
    "trg_slf_attn_pre_softmax_shape": (2L, ),
    "trg_slf_attn_post_softmax_shape": (4L, ),
    "trg_src_attn_pre_softmax_shape": (2L, ),
    "trg_src_attn_post_softmax_shape": (4L, ),
    "enc_output":
    (1, (ModelHyperParams.max_length + 1), ModelHyperParams.d_model),
    "lbl_word": (1 * (ModelHyperParams.max_length + 1), 1L),
    "lbl_weight": (1 * (ModelHyperParams.max_length + 1), 1L),
}
