import ctypes
import numpy
from ctypes import cdll
from numpy import ndarray
from numpy import half
from numpy import int32

# TODO: 载入动态连接库 libst.so，为之后调用 libst.so 定义的函数做准备
lib = cdll.LoadLibrary("./bert_lib/libst.so")

lib.bertAllocate.restype = ctypes.c_void_p
u16_p = ctypes.POINTER(ctypes.c_uint16)
# TODO：#定义 ctypes 指针类型 i32_p，指向 C 语言中的 c_int32 类型
i32_p = ctypes.POINTER(ctypes.c_int32)
type_map = {
    numpy.float16: u16_p,
    half: u16_p,
    int32: i32_p,
}

'''
gets the data pointer of the ndarray.
converts dtype to a temp ndarray if necessary
'''


def to_ptr(arr: ndarray, type=half):
    if(arr.dtype == type):
        return arr.ctypes.data_as(type_map[type])
    casted = arr.astype(type)
    return casted.ctypes.data_as(type_map[type])


'''
converts list of ndarray to a pointer (c-style raw array).
each element is the data pointer of corresponding ndarray.
'''


def wrap_ndarray_list(l: 'list[ndarray]', type=numpy.half):
    temp = [to_ptr(item, type) for item in l]
    arr = (u16_p * len(temp))(*temp)
    return arr


def check_list(l: 'list[ndarray]', length: int, shape: 'list[int]' = None):
    if(len(l) != length):
        raise ValueError("length of list is not expected!")

    if shape != None:
        for a in l:
            if(list(a.shape) != shape):
                raise ValueError("ndarray shape mismatch")


def check_shape(a: ndarray, shape: 'list[int]'):
    if(list(a.shape) != shape):
        raise ValueError("ndarray shape mismatch")


def extract(weights: 'list[map[str, map[str, ndarray]]]', key0: str, key1: str) -> 'list[ndarray]':
    return wrap_ndarray_list([w[key0][key1] for w in weights], half)

def read_data(file_path, file, row, col, ddtype = numpy.half):
    data = numpy.fromfile(file_path + file, ddtype, count=-1, sep="\n")
    data = data.reshape((row, col))
    return data

def computeMask(file):
    exist = (file > 0) * 1.0
    factor = numpy.ones(file.shape[1])
    res = numpy.dot(exist, factor)
    #res = res.reshape(-1, 1)
    return res

def diff(fa, fb):
    num1 = fa.shape[0]
    num2 = fb.shape[0]
    if (num1 != num2):
        print("error!, file2len ! len fileb")
    sumt = 0.
    sumd = 0.
    for i in range(num1):
        sumt += abs(fb[i])
        d = abs(fa[i] - fb[i])
        sumd += d
    mse = sumd/sumt
    print('diff is %f' %mse)

def get_encoder_weight(example_encoder_weights, encoder_layer, path_weight):
    for i in range(encoder_layer):
        query_weight = "bert.encoder.layer." + str(i) + ".attention.self.query.weight"

        example_encoder_weights[i]["self_attn"]["q_weight"] = read_data(path_weight, query_weight,
                                                              example_encoder_weights[i]["self_attn"]["q_weight"].shape[0],
                                                              example_encoder_weights[i]["self_attn"]["q_weight"].shape[1])
        key_weight = "bert.encoder.layer." + str(i) + ".attention.self.key.weight"
        example_encoder_weights[i]["self_attn"]["k_weight"] = read_data(path_weight, key_weight,
                                                              example_encoder_weights[i]["self_attn"]["k_weight"].shape[0],
                                                              example_encoder_weights[i]["self_attn"]["k_weight"].shape[1])
        value_weight = "bert.encoder.layer." + str(i) + ".attention.self.value.weight"
        example_encoder_weights[i]["self_attn"]["v_weight"] = read_data(path_weight, value_weight,
                                                              example_encoder_weights[i]["self_attn"]["v_weight"].shape[0],
                                                              example_encoder_weights[i]["self_attn"]["v_weight"].shape[1])
        query_bias = "bert.encoder.layer." + str(i) + ".attention.self.query.bias"
        example_encoder_weights[i]["self_attn"]["q_bias"] = numpy.fromfile(path_weight + query_bias, numpy.half, count=-1, sep="\n")
        key_bias = "bert.encoder.layer." + str(i) + ".attention.self.key.bias"
        example_encoder_weights[i]["self_attn"]["k_bias"] = numpy.fromfile(path_weight + key_bias, numpy.half, count=-1, sep="\n")
        value_bias = "bert.encoder.layer." + str(i) + ".attention.self.value.bias"
        example_encoder_weights[i]["self_attn"]["v_bias"] = numpy.fromfile(path_weight + value_bias, numpy.half, count=-1, sep="\n")

        self_attn_layernormal_scale = "bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.weight"
        self_attn_layernormal_bias = "bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.bias"
        self_attn_output_weight = "bert.encoder.layer." + str(i) + ".attention.output.dense.weight"
        self_attn_output_bias = "bert.encoder.layer." + str(i) + ".attention.output.dense.bias"

        example_encoder_weights[i]["self_attn"]["layernormal_scale"] = numpy.fromfile(path_weight + self_attn_layernormal_scale, numpy.half, count=-1, sep="\n")
        example_encoder_weights[i]["self_attn"]["layernormal_bias"] = numpy.fromfile(path_weight + self_attn_layernormal_bias, numpy.half, count=-1, sep="\n")
        example_encoder_weights[i]["self_attn"]["out_bias"] = numpy.fromfile(path_weight + self_attn_output_bias, numpy.half, count=-1, sep="\n")
        example_encoder_weights[i]["self_attn"]["out_weight"] = read_data(path_weight, self_attn_output_weight,
                                                              example_encoder_weights[i]["self_attn"]["out_weight"].shape[0],
                                                              example_encoder_weights[i]["self_attn"]["out_weight"].shape[1])

        ffn_inner_weight = "bert.encoder.layer." + str(i) + ".intermediate.dense.weight"
        ffn_inner_bias = "bert.encoder.layer." + str(i) + ".intermediate.dense.bias"
        ffn_outer_weight = "bert.encoder.layer." + str(i) + ".output.dense.weight"
        ffn_outer_bias = "bert.encoder.layer." + str(i) + ".output.dense.bias"
        ffn_layernormal_scale = "bert.encoder.layer." + str(i) + ".output.LayerNorm.weight"
        ffn_layernormal_bias = "bert.encoder.layer." + str(i) + ".output.LayerNorm.bias"
        example_encoder_weights[i]["ffn"]["inner_weight"] = read_data(path_weight, ffn_inner_weight,
                                                              example_encoder_weights[i]["ffn"]["inner_weight"].shape[0],
                                                              example_encoder_weights[i]["ffn"]["inner_weight"].shape[1])

        example_encoder_weights[i]["ffn"]["out_weight"] = read_data(path_weight, ffn_outer_weight,
                                                              example_encoder_weights[i]["ffn"]["out_weight"].shape[0],
                                                              example_encoder_weights[i]["ffn"]["out_weight"].shape[1])
        example_encoder_weights[i]["ffn"]["inner_bias"] = numpy.fromfile(path_weight + ffn_inner_bias, numpy.half, count=-1, sep="\n")
        example_encoder_weights[i]["ffn"]["out_bias"] = numpy.fromfile(path_weight + ffn_outer_bias, numpy.half, count=-1, sep="\n")
        example_encoder_weights[i]["ffn"]["layernormal_scale"] = numpy.fromfile(path_weight + ffn_layernormal_scale, numpy.half, count=-1, sep="\n")
        example_encoder_weights[i]["ffn"]["layernormal_bias"] = numpy.fromfile(path_weight + ffn_layernormal_bias, numpy.half, count=-1, sep="\n")

class PyBert:
    max_batch_size = 64
    max_seq_len = 512  # 1 to 512
    head_size = 64  # must be 64
    head_num = 12  # must be 12
    hidden = 768  # must be 768
    ffn_hidden = 3072  # must be 3072
    hidden_act = 8 #"gelu"
    def __init__(self):
        # 64 batch is the maximum supported size of cnnl_extra operators
        lib.bertInit()

    def __del__(self):
        lib.bertDestroy(ctypes.c_void_p(self.ptr))

    def allocate(self, dictionary_len: int, type_dictionary_len: int, encoder_layers: int, eos_id: int, bos_id: int):
        self.dictionary_len = dictionary_len
        self.type_dictionary_len = type_dictionary_len
        self.encoder_layers = encoder_layers

        #TODO:调用PyBert中的参数，作为bertAllocate 的函数参数
        max_batch_size = PyBert.max_batch_size
        max_seq_len = PyBert.max_seq_len
        head_size = PyBert.head_size
        head_num = PyBert.head_num
        ffn_hidden = PyBert.ffn_hidden
        hidden_act = PyBert.hidden_act
        calc_type           = 1 
        filter_type         = 1 

        #TODO: 调用libst.so的bertAllocate 函数以及上述一系列参数，完成bert 模型在 MLU内存上的分配，并将 bertAllocate 返回的分配地址传递给 self.ptr
        self.ptr = lib.bertAllocate(max_batch_size, 
        max_seq_len, 
        head_size, 
        head_num, 
        dictionary_len,
        type_dictionary_len,
        ffn_hidden, 
        hidden_act, 
        encoder_layers,
        eos_id,
        bos_id,
        calc_type, 
        filter_type)

    # ndarrays dtype should be half.
    def load_weight(
        self,
        bert_pre_weights: 'map[str, ndarray]',
        encoder_layers_weights: 'list[map[str, map[str, ndarray]]]',
        bert_post_weights: 'map[str, ndarray]'
    ):
        check_shape(bert_pre_weights["bert_pre_dict"], [self.dictionary_len, PyBert.hidden])
        check_shape(bert_pre_weights["bert_pre_type_dict"], [self.type_dictionary_len, PyBert.hidden])
        check_shape(bert_pre_weights["bert_pre_position_weight"], [PyBert.max_seq_len, PyBert.hidden])
        assert(len(encoder_layers_weights) == self.encoder_layers)
        
        

        ##  指定 bertLoadfilter 的函数参数
        obj                        = ctypes.c_void_p(self.ptr)
        bert_pre_dict              = to_ptr(bert_pre_weights["bert_pre_dict"])
        bert_pre_type_dict         = to_ptr(bert_pre_weights["bert_pre_type_dict"])
        bert_pre_position_filter   = to_ptr(bert_pre_weights["bert_pre_position_weight"])
        bert_pre_layernormal_scale = to_ptr(bert_pre_weights["bert_pre_layernormal_scale"])
        bert_pre_layernormal_bias  = to_ptr(bert_pre_weights["bert_pre_layernormal_bias"])

        encoder_self_attn_q_filter_layers           = extract(encoder_layers_weights, "self_attn", "q_weight")
        encoder_self_attn_q_bias_layers             = extract(encoder_layers_weights, "self_attn", "q_bias")
        encoder_self_attn_k_filter_layers           = extract(encoder_layers_weights, "self_attn", "k_weight")
        encoder_self_attn_k_bias_layers             = extract(encoder_layers_weights, "self_attn", "k_bias")
        encoder_self_attn_v_filter_layers           = extract(encoder_layers_weights, "self_attn", "v_weight")
        encoder_self_attn_v_bias_layers             = extract(encoder_layers_weights, "self_attn", "v_bias")
        encoder_self_attn_out_filter_layers         = extract(encoder_layers_weights, "self_attn", "out_weight")
        encoder_self_attn_out_bias_layers           = extract(encoder_layers_weights, "self_attn", "out_bias")
        encoder_self_attn_layernormal_filter_layers = extract(encoder_layers_weights, "self_attn", "layernormal_scale")
        encoder_self_attn_layernormal_bias_layers   = extract(encoder_layers_weights, "self_attn", "layernormal_bias")

        encoder_ffn_inner_filter_layers      = extract(encoder_layers_weights, "ffn", "inner_weight")
        encoder_ffn_inner_bias_layers        = extract(encoder_layers_weights, "ffn", "inner_bias")
        encoder_ffn_out_filter_layers        = extract(encoder_layers_weights, "ffn", "out_weight")
        encoder_ffn_out_bias_layers          = extract(encoder_layers_weights, "ffn", "out_bias")
        encoder_ffn_layernormal_scale_layers = extract(encoder_layers_weights, "ffn", "layernormal_scale")
        encoder_ffn_layernormal_bias_layers  = extract(encoder_layers_weights, "ffn", "layernormal_bias")

        bert_post_filter = to_ptr(bert_post_weights["bert_post_weight"])
        bert_post_bias   = to_ptr(bert_post_weights["bert_post_bias"])

        # TODO: 调用 bertLoadfilter 函数，完成参数权值的加载
        lib.bertLoadfilter(obj, 
        bert_pre_dict, 
        bert_pre_type_dict, 
        bert_pre_position_filter, 
        bert_pre_layernormal_scale, 
        bert_pre_layernormal_bias, 

        encoder_self_attn_q_filter_layers, 
        encoder_self_attn_q_bias_layers, 
        encoder_self_attn_k_filter_layers, 
        encoder_self_attn_k_bias_layers, 
        encoder_self_attn_v_filter_layers, 
        encoder_self_attn_v_bias_layers, 
        encoder_self_attn_out_filter_layers, 
        encoder_self_attn_out_bias_layers, 
        encoder_self_attn_layernormal_filter_layers, 
        encoder_self_attn_layernormal_bias_layers, 

        encoder_ffn_inner_filter_layers, 
        encoder_ffn_inner_bias_layers, 
        encoder_ffn_out_filter_layers, 
        encoder_ffn_out_bias_layers, 
        encoder_ffn_layernormal_scale_layers,
        encoder_ffn_layernormal_bias_layers,
        
        bert_post_filter, 
        bert_post_bias)

    def forward(
        self,
        tokens: ndarray,  # [batch, seq_len], dtype=int32
        type_tokens: ndarray, # [batch, seq_len], dtype=int32
        attention_mask: ndarray, # [batch], dtype=int32
        dump_file_path: str = None
    ):
        actual_batch = tokens.shape[0]
        seq_len = tokens.shape[1]
        type_tokens_batch = type_tokens.shape[0]
        type_tokens_seq = type_tokens.shape[1]
        hwtime = ctypes.c_float()
        fwtime = ctypes.c_float()
        seq_output = ndarray([actual_batch, seq_len, 2], half)
        pool_output = ndarray([actual_batch, 2], half)

        

        # 指定 bertForward 的函数参数
        obj               = ctypes.c_void_p(self.ptr)
        tokens            = to_ptr(tokens, int32)
        actual_seq_len    = seq_len
        type_tokens       = to_ptr(type_tokens, int32)
        attn_mask         = to_ptr(attention_mask, int32)
        dump_dir          = ctypes.c_char_p(dump_file_path.encode("utf-8"))

        #TODO：调用 bertForward，完成单次推理
        lib.bertForward(obj, 
        tokens, 
        actual_batch, 
        actual_seq_len, 
        type_tokens, 
        type_tokens_batch,
        type_tokens_seq,
        attn_mask, 
        to_ptr(seq_output), 
        to_ptr(pool_output), 
        ctypes.byref(hwtime), 
        ctypes.byref(fwtime), 
        dump_dir)


        # cnrtSync is called inside Bert_Forward
        return seq_output,pool_output,hwtime.value,fwtime.value

