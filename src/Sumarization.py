import argparse
from vncorenlp import VnCoreNLP
import time
import os
from others.logging import init_logger
from prepro import data_builder
from train_abstractive import validate_abs, test_abs

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def do_format_to_bert(args):
    print(time.clock())
    args.mode = "format_to_bert"
    data_builder.format_to_bert(args)
    print(time.clock())

def do_format_to_json(args):
    rdrsegmenter = VnCoreNLP("VietnameseAbsSumarization/src/vncorenlp/VnCoreNLP-1.1.1.jar",
                             annotators="wseg", max_heap_size='-Xmx500m')
    count = 0
    count_json_file = 0
    files = os.listdir(args.raw_path)
    content_json = "["
    for file_path in files:
        try:
            file = open(args.raw_path + file_path)
            content_file = file.read()
            lines = []
            segmentations = rdrsegmenter.tokenize(content_file)
            for seg in segmentations:
                lines.append(" ".join(seg))
            print(lines)
            content_json += """\n{\n"src":["""
            content_format_json = list()
            highlight_format_json = list()
            index_highlight = 0
            for i in range(len(lines)):
                line = lines[i].replace("\n", "")
                if line == "":
                    index_highlight = i + 1
                    break

                words = []
                for word in line.split(" "):
                    if "\"" in word:
                        word = word.replace("\"", "\\\"")
                    words.append(word)

                content_format_json.append("[\n" + ",\n".join(["\"" + word + "\"" for word in words]) + "\n]")

            for line in lines[index_highlight:]:
                line = line.replace("\n", "")
                if line == "@highlight" or line == "":
                    continue

                words = []
                for word in line.split(" "):
                    if "\"" in word:
                        word = word.replace("\"", "\\\"")
                    words.append(word)
                highlight_format_json.append("[\n" + ",\n".join(["\"" + word + "\"" for word in words]) + "\n]")

            content_json += ",\n".join(content_format_json) + """\n],\n"tgt": [\n""" + ",\n".join(
                highlight_format_json) + "\n]\n},"

            if count == 0 and count % 1000 == 0:
                content_json += "]"
                index_last_comma = content_json.rfind(",")
                content_json = content_json[:index_last_comma] + content_json[index_last_comma + 1:]
                file_save = open(args.raw_path + "Vietnamese.test." + str(count_json_file) + ".json", "w")
                file_save.write(content_json)
                content_json = "["
                count_json_file += 1
            count += 1
        except Exception:
            continue

parser = argparse.ArgumentParser()
parser.add_argument("-pretrained_model", default='bert', type=str)
parser.add_argument("-mode", default='', type=str)
parser.add_argument("-select_mode", default='greedy', type=str)
parser.add_argument("-map_path", default='../../data/')
parser.add_argument("-raw_path", default='/content/TestData/')
parser.add_argument("-save_path", default='/content/PtDataForPreSumm/')
parser.add_argument("-shard_size", default=2000, type=int)
parser.add_argument('-min_src_nsents', default=3, type=int)
parser.add_argument('-max_src_nsents', default=100, type=int)
parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
parser.add_argument('-min_tgt_ntokens', default=5, type=int)
parser.add_argument('-max_tgt_ntokens', default=500, type=int)
parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument('-log_file', default='../../logs/cnndm.log')
parser.add_argument('-dataset', default='')
parser.add_argument('-n_cpus', default=1, type=int)
args = parser.parse_args()

parser1 = argparse.ArgumentParser()
parser1.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
parser1.add_argument("-encoder", default='baseline', type=str, choices=['bert', 'baseline'])
parser1.add_argument("-mode", default='test', type=str, choices=['train', 'validate', 'test'])
parser1.add_argument("-bert_data_path", default=args.save_path + "Vietnamese")
parser1.add_argument("-model_path", default='/content/drive/MyDrive/Colab Notebooks/Models/PreSumm/')
parser1.add_argument("-result_path", default='VietnameseAbsSumarization/logs/abs_bert_cnndm')
parser1.add_argument("-temp_dir", default='VietnameseAbsSumarization/logs/')

parser1.add_argument("-batch_size", default=300, type=int)
parser1.add_argument("-test_batch_size", default=50, type=int)

parser1.add_argument("-max_pos", default=512, type=int)
parser1.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
parser1.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
parser1.add_argument("-load_from_extractive", default='', type=str)

parser1.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=True)
parser1.add_argument("-lr_bert", default=2e-3, type=float)
parser1.add_argument("-lr_dec", default=2e-3, type=float)
parser1.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

parser1.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
parser1.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
parser1.add_argument("-dec_dropout", default=0.2, type=float)
parser1.add_argument("-dec_layers", default=6, type=int)
parser1.add_argument("-dec_hidden_size", default=768, type=int)
parser1.add_argument("-dec_heads", default=8, type=int)
parser1.add_argument("-dec_ff_size", default=2048, type=int)
parser1.add_argument("-enc_hidden_size", default=512, type=int)
parser1.add_argument("-enc_ff_size", default=512, type=int)
parser1.add_argument("-enc_dropout", default=0.2, type=float)
parser1.add_argument("-enc_layers", default=6, type=int)

# params for EXT
parser1.add_argument("-ext_dropout", default=0.2, type=float)
parser1.add_argument("-ext_layers", default=2, type=int)
parser1.add_argument("-ext_hidden_size", default=768, type=int)
parser1.add_argument("-ext_heads", default=8, type=int)
parser1.add_argument("-ext_ff_size", default=2048, type=int)

parser1.add_argument("-label_smoothing", default=0.1, type=float)
parser1.add_argument("-generator_shard_size", default=32, type=int)
parser1.add_argument("-alpha",  default=0.9, type=float)
parser1.add_argument("-beam_size", default=5, type=int)
parser1.add_argument("-min_length", default=20, type=int)
parser1.add_argument("-max_length", default=100, type=int)
parser1.add_argument("-max_tgt_len", default=140, type=int)



parser1.add_argument("-param_init", default=0, type=float)
parser1.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
parser1.add_argument("-optim", default='adam', type=str)
parser1.add_argument("-lr", default=1, type=float)
parser1.add_argument("-beta1", default= 0.9, type=float)
parser1.add_argument("-beta2", default=0.999, type=float)
parser1.add_argument("-warmup_steps", default=8000, type=int)
parser1.add_argument("-warmup_steps_bert", default=8000, type=int)
parser1.add_argument("-warmup_steps_dec", default=8000, type=int)
parser1.add_argument("-max_grad_norm", default=0, type=float)

parser1.add_argument("-save_checkpoint_steps", default=5, type=int)
parser1.add_argument("-accum_count", default=1, type=int)
parser1.add_argument("-report_every", default=1, type=int)
parser1.add_argument("-train_steps", default=1000, type=int)
parser1.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


parser1.add_argument('-visible_gpus', default='0', type=str)
parser1.add_argument('-gpu_ranks', default='0', type=str)
parser1.add_argument('-log_file', default='../logs/cnndm.log')
parser1.add_argument('-seed', default=666, type=int)

parser1.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
parser1.add_argument("-test_from", default='')
parser1.add_argument("-test_start_from", default=-1, type=int)

parser1.add_argument("-train_from", default='')
parser1.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
parser1.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
args1 = parser1.parse_args()

do_format_to_json(args)
do_format_to_bert(args)
device = "cpu" if args1.visible_gpus == '-1' else "cuda"
device_id = 0 if device == "cuda" else -1
args1.gpu_ranks = [int(i) for i in range(len(args1.visible_gpus.split(',')))]
args1.world_size = len(args1.gpu_ranks)
# validate_abs(args1, device_id)
cp = args1.test_from
try:
    step = int(cp.split('.')[-2].split('_')[-1])
except:
    step = 0
test_abs(args, device_id, cp, step)