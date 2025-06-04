import ctypes
import numpy
import numpy as np
from ctypes import cdll

from numpy import ndarray
from numpy import half
from numpy import int32

import argparse 
import logging
import os
import time
from tqdm import tqdm, trange
import torch

from utils.utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,SquadDatasetLoader,
                         RawResultExtended, write_predictions_extended, InputFeatures)

from utils.tokenization_bert import (BertTokenizer)

from bert_lib.bert import (PyBert,read_data,get_encoder_weight,computeMask)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
from utils.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

path_weight = './bert_model/'

logger = logging.getLogger(__name__)

def saveResult(imageNum,batch_size,throughput, top1, top5):
    if not os.getenv('OUTPUT_JSON_FILE'):
        return
    TIME=-1
    e2eFps=throughput
    latency=round(float((1000)/float(e2eFps))*float(batch_size),2)
    result={
            "Output":{
                "Accuracy":{
                    "top1":'%.2f'%top1,
                    "top5":'%.2f'%top5,
                    "f1":'%.2f'% -1,
                    "exact_match":'%.2f'% -1,
                    },
                "HostLatency(ms)":{
                    "average":'%.2f'%latency,
                    "throughput(fps)":'%.2f'%e2eFps,
                    }
                }
            }
    with open(os.getenv("OUTPUT_JSON_FILE"),"a") as outputfile:
        json.dump(result,outputfile,indent=4,sort_keys=True)
        outputfile.write('\n')
        outputfile.close()

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_file = args.predict_file
    logger.info("Creating features from dataset file at %s", input_file)
    #TODO: 调用utils.utils_squad中的函数读取 SQuAD 格式测例
    examples = read_squad_examples(input_file=input_file,
                                            is_training=not evaluate,
                                            version_2_with_negative=args.version_2_with_negative)
    #TODO: 调用utils.utils_squad函数将SQuAD测例转换为特征表示
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            doc_stride=args.doc_stride,
                                            max_query_length=args.max_query_length,
                                            is_training=not evaluate)

    # MLU requires number of features must be a multiple of batch_size,
    # so we pad with fake features which are ignored later on.
    def pad_fake_feature():
        features.append(
            InputFeatures(
                unique_id=0,
                example_index=0,
                doc_span_index=0,
                tokens=features[-1].tokens,
                token_to_orig_map=features[-1].token_to_orig_map,
                token_is_max_context=features[-1].token_is_max_context,
                input_ids=[0] * args.max_seq_length,
                input_mask=[0] * args.max_seq_length,
                segment_ids=[0] * args.max_seq_length,
                cls_index=0,
                p_mask=[0] * args.max_seq_length,
                paragraph_len=0,
                start_position=None,
                end_position=None,
                is_impossible=False))

    #TODO: 当特征数量不是 batch_size 的整倍数时，执行以下循环
    while len(features) % args.batch_size != 0:
        #TODO: 调用函数以使用虚假特征进行填充
        pad_fake_feature()
        logger.info("  Pad one feature to eval_features, num of eval_features is %d now.", len(features))

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    #TODO: 从 features 列表中提取 input_mask 属性，形成一个列表
    all_input_mask = [f.input_mask for f in features]
    #TODO: 从 features 列表中提取 segment_ids 属性，形成一个列表
    all_segment_ids = [f.segment_ids for f in features]
    #TODO: 从 features 列表中提取 cls_index 属性，形成一个列表
    all_cls_index = [f.cls_index for f in features]
    #TODO: 从 features 列表中提取 p_mask 属性，形成一个列表
    all_p_mask = [f.p_mask for f in features]
        
    #TODO: 将之前创建的五个列表转换为 NumPy 数组
    all_input_ids = np.array(all_input_ids)
    all_input_mask = np.array(all_input_mask)
    all_segment_ids = np.array(all_segment_ids) 
    all_cls_index = np.array(all_cls_index)
    all_p_mask = np.array(all_p_mask) 
    if evaluate:
        all_example_index = np.arange(len(features))
        dataset = (all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset

def get_weights(dict_len, type_dict_len):

    hidden: int = PyBert.hidden
    ffn_hidden: int = PyBert.ffn_hidden
    max_seq_len: int = PyBert.max_seq_len

    # bert_pre
    example_bert_pre_weights: 'map[str, ndarray]' = {
        "bert_pre_dict": ndarray([dict_len, hidden], half),
        "bert_pre_type_dict": ndarray([type_dict_len, hidden], half),
        "bert_pre_position_weight": ndarray([max_seq_len, hidden], half),
        "bert_pre_layernormal_scale": ndarray([hidden, hidden], half),
        "bert_pre_layernormal_bias": ndarray([hidden], half)
    }
    # 12 layers encoder
    example_encoder_weights: 'list[map[str, map[str, ndarray]]]' = [
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        },
        {
            "self_attn": {
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "q_weight": ndarray([hidden, hidden], half),
                "k_weight": ndarray([hidden, hidden], half),
                "v_weight": ndarray([hidden, hidden], half),
                "q_bias": ndarray([hidden], half),
                "k_bias": ndarray([hidden], half),
                "v_bias": ndarray([hidden], half),
                "out_weight": ndarray([hidden, hidden], half),
                "out_bias": ndarray([hidden], half)
            },
            "ffn":{
                "layernormal_scale": ndarray([hidden], half),
                "layernormal_bias": ndarray([hidden], half),
                "inner_weight": ndarray([ffn_hidden, hidden], half),
                "inner_bias": ndarray([ffn_hidden], half),
                "out_weight": ndarray([hidden, ffn_hidden], half),
                "out_bias": ndarray([hidden], half)
            }
        }
    ]
    # bert_post
    example_bert_post_weights: 'map[str, ndarray]' = {
        "bert_post_weight": ndarray([2, hidden], half),
        "bert_post_bias": ndarray([2], half)
        # "bert_post_layernormal_scale": ndarray([hidden, hidden], half),
        # "bert_post_layernormal_bias": ndarray([hidden], half),
    }
    example_bert_pre_weights["bert_pre_dict"] = read_data(path_weight, "bert.embeddings.word_embeddings.weight",
                                                example_bert_pre_weights["bert_pre_dict"].shape[0],
                                                example_bert_pre_weights["bert_pre_dict"].shape[1])

    example_bert_pre_weights["bert_pre_type_dict"] = read_data(path_weight, "bert.embeddings.token_type_embeddings.weight",
                                                     example_bert_pre_weights["bert_pre_type_dict"].shape[0],
                                                     example_bert_pre_weights["bert_pre_type_dict"].shape[1])
    example_bert_pre_weights["bert_pre_position_weight"] = read_data(path_weight, "bert.embeddings.position_embeddings.weight",
                                                           example_bert_pre_weights["bert_pre_position_weight"].shape[0],
                                                           example_bert_pre_weights["bert_pre_position_weight"].shape[1])

    example_bert_pre_weights["bert_pre_layernormal_scale"] = numpy.fromfile(path_weight + "bert.embeddings.LayerNorm.weight",
                                                               numpy.half, count=-1, sep="\n")

    example_bert_pre_weights["bert_pre_layernormal_bias"] = numpy.fromfile(path_weight + "bert.embeddings.LayerNorm.bias",
                                                               numpy.half, count=-1, sep="\n")

    example_bert_post_weights["bert_post_weight"] = read_data(path_weight, "qa_outputs.weight",
                                                           example_bert_post_weights["bert_post_weight"].shape[0],
                                                           example_bert_post_weights["bert_post_weight"].shape[1])

    example_bert_post_weights["bert_post_bias"] = numpy.fromfile(path_weight + "qa_outputs.bias",
                                                               numpy.half, count=-1, sep="\n")


    get_encoder_weight(example_encoder_weights, 12, path_weight)
    return example_bert_pre_weights, example_encoder_weights, example_bert_post_weights
    
def evaluate(args, tokenizer, prefix=""):
    # TODO: 调用数据加载模块加载数据集
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.batch_size
    # TODO: 调用函数用以创建数据加载器来加载数据
    eval_dataloader = SquadDatasetLoader(dataset, len(features), batch_size=args.eval_batch_size)
    print("Load SQuAD data Pass!")

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(features))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    # model

    dict_len: int = 30522
    type_dict_len: int = 2
    encoder_layers: int = 12
    eos_id: int = 1
    bos_id: int = 0

    bert = PyBert()
    bert.allocate(dict_len, type_dict_len, 12, 1, 0)
    #TODO: 调用函数以获取预训练模型的权重
    example_bert_pre_weights, example_encoder_weights, example_bert_post_weights = get_weights(dict_len, type_dict_len)
    logger.info("Bert: loading weights...")
    #TODO: 调用PyBert中的函数以加载权重到模型
    bert.load_weight(example_bert_pre_weights, example_encoder_weights, example_bert_post_weights)
    print("Load Model Pass!")
    logger.info("Bert: Evaluating")

    start = time.time()
    for sample in tqdm(eval_dataloader, desc="Evaluating"):
        sample = tuple(t for t in sample)
        tokens          = sample[0]     # all_input_ids ndarray,  [batch, seq_len], dtype=int32
        attention_mask  = sample[1]     # all_input_mask ndarray,  [batch, seq_len], dtype=int32
        type_tokens     = sample[2]     
        example_indices = sample[3]

        attention_mask = computeMask(attention_mask)

        #TODO: 调用PyBert 模型的前向传播中cnnl大算子完成推理
        logits, pooler_out, hwtime, fwtime  = bert.forward(tokens, type_tokens, attention_mask, "/workspace/cambricon_demo/Bert")
        #TODO: 将模型输出的 logits 转换为浮点数
        logits=logits.astype(float)
        start_logits, end_logits = np.split(logits, 2,axis=2)
        start_logits = np.squeeze(start_logits, axis=2)
        end_logits = np.squeeze(end_logits, axis=2)
        outputs = (start_logits, end_logits,)
        out0 = outputs[0]
        out1 = outputs[1]
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id    = unique_id,
                               start_logits = list(out0[i]),
                               end_logits   = list(out1[i]))
            #TODO: 将当前结果添加到 all_results 列表中
            all_results.append(result)
    end = time.time()
    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    write_predictions(examples, features, all_results, args.n_best_size,
                    args.max_answer_length, args.do_lower_case, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                    args.version_2_with_negative, args.null_score_diff_threshold)
    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    # TODO: 调用utils.utils_squad_evaluate中的函数对推理结果进行评估
    results = evaluate_on_squad(evaluate_options)
    logger.info("E2E time: %.3f s", float(end-start))
    logger.info("throughput: %.3f", (float(len(features))/float((end-start))))
    #dump result to Json
    saveResult(len(features),args.batch_size,len(features)/float((end-start)), -1, -1)
    print("The infer module Pass!")
    return results




def main():
    #TODO: # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    ## Required parameters
    #TODO：给一个解析器添加程序参数信息
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')

    parser.add_argument("--do_lower_case", action='store_false',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")

    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")

    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--do_eval", default=True,
                        help="Whether to run eval on the dev set.")

    #TODO: 解析命令行参数
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    global_step = ""
    #TODO: 调用模型评估函数进行评估
    result = evaluate(args, tokenizer, prefix=global_step)
    result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())

    logger.info("Results: {}".format(result))


# usage example
if __name__ == "__main__":
    main()
