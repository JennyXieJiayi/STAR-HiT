'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of STAR-HiT model in:
Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation
'''
import argparse
import time
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--train_num_neg', type=int, default=5000,
                        help='The number of negative samples for sampled softmax')
    parser.add_argument('--test_num_neg', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--opt_factor', type=int, default=1)
    parser.add_argument('--opt_warmup', type=int, default=400)
    parser.add_argument('--print_every', type=int, default=4,
                        help='Iteration interval of printing loss.')
    parser.add_argument('--save_every', type=int, default=4,
                        help='Iteration interval of saving model.')
    parser.add_argument('--evaluate_every', type=int, default=4,
                        help='Epoch interval of evaluation.')
    parser.add_argument('--head_num', type=int, default=4, choices=[1,2,4,8])
    parser.add_argument('--layer_stack_num', type=int, default=2, choices=[1,2,3])
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--hid_size', type=int, default=128)
    parser.add_argument('--sub_seq_len', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--pre_model_path', type=str, default='')
    parser.add_argument('--data_root', type=str, default='./datasets/FoursquareNYC')

    args = parser.parse_args()

    data_meta_path = os.path.join(args.data_root, 'data_info.csv')
    args.user_vocab, args.poi_vocab, args.poi_maxlen = pd.read_csv(data_meta_path, sep='\t').values.squeeze()
    args.user_vocab += 1
    args.poi_vocab += 1

    args.data_name = args.data_root.split(os.path.sep)[-1]
    save_dir = 'models/{}/{}/'.format(
        args.data_name, time.strftime("%Y%m%d_%H%M%S"))
    args.save_dir = save_dir

    return args