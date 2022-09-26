'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of STAR-HiT model in:
Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation
'''
from utils.parser import *
from utils.utils import *
from utils.log_helper import *
from metrics import *
from data import *
from model import *
import torch
import numpy as np
import pandas as pd
import random
import logging


def predict(args, phase='test'):
	# seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# log
	log_save_id = create_log_id(args.trained_model_path)
	logging_config(folder=args.trained_model_path, name='log{:d}'.format(log_save_id), no_console=False)
	logging.info(args)

	# GPU / CPU
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:{}".format(args.cuda_idx) if torch.cuda.is_available() else "cpu")

	# load model
	model = STARHiT(poi_vocab=args.poi_vocab,
	                poi_maxlen=args.poi_maxlen,
	                emb_size=args.emb_size,
	                hid_size=args.hid_size,
	                head_num=args.head_num,
	                block_num=args.layer_stack_num,
	                sub_seq_len=args.sub_seq_len,
	                dropout=args.dropout)

	trained_model = load_model(model, args.trained_model_file)

	# load data
	test_data = NextPOIDataset(phase=phase,
	                           poi_vocab=args.poi_vocab,
                               data_root=args.data_root,
                               poi_maxlen=args.poi_maxlen,
                               logging=logging,
                               test_num_neg=args.test_num_neg)
	hits, _, ndcgs = evaluate(trained_model, test_data, args.test_batch_size, [5,10], use_cuda, device, args.test_num_neg)

	return hits, ndcgs


if __name__ == "__main__":
	args = parse_args()
	trained_model_path = './models/FoursquareNYC'
	model_name = 'model_2.pth'
	args.trained_model_path = trained_model_path
	args.trained_model_file = os.path.join(trained_model_path, model_name)
	hits, ndcgs = predict(args)
	print([model_name, hits, ndcgs])
	test_metrics = pd.DataFrame([model_name, hits, ndcgs]).transpose()
	test_metrics.to_csv(os.path.join(trained_model_path, 'test_results.csv'), mode='a', header=['model', 'hit@5, hit@10', 'ndcg@5, ndcg@10'], sep='\t', index=False)
