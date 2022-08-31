'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of STAR-HiT model in:
Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation
'''
from utils.parser import *
from utils.log_helper import *
from utils.optimizer import NoamOpt
from utils.utils import *
from metrics import *
from data import *
from model import *
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import random
import logging
import os


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # log
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    if args.use_cuda:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:{}".format(args.cuda_idx) if torch.cuda.is_available() else "cpu")
    else:
        use_cuda = False
        device = torch.device('cpu')

    # load data
    data = NextPOIDataset(phase='train',
                          data_root=args.data_root,
                          poi_maxlen=args.poi_maxlen,
                          logging=logging)
    data_loader = DataLoader(data,
                      batch_size=args.train_batch_size,
                      shuffle=True)
    batch_num=len(data_loader)

    # construct model
    model = STARHiT(poi_vocab=args.poi_vocab,
                    poi_maxlen=args.poi_maxlen,
                    emb_size=args.emb_size,
                    hid_size=args.hid_size,
                    head_num=args.head_num,
                    block_num=args.layer_stack_num,
                    sub_seq_len=args.sub_seq_len,
                    dropout=args.dropout)
    model.to(device)
    logging.info(model)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if os.path.isfile(args.pre_model_path):
        model = load_model(model, args.pre_model_path)
        logging.info("Pre-trained model: {}".format(args.pre_model_path))

    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = NoamOpt(args.emb_size, args.opt_factor, args.opt_warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    logging.info(optimizer)

    # initialize metrics
    init_metrics = pd.DataFrame(['epoch_idx', ['hit@5', 'hit@10'], ['ndcg@5', 'ndcg@10']]).transpose()
    init_metrics.to_csv(os.path.join(args.save_dir, 'train_results.csv'), mode='a', header=False, sep='\t',
                   index=False)
    best_epoch = -1
    last_save_epoch = -1
    ndcg5_max = -np.inf
    ndcg10_cor = -np.inf
    hit5_cor = -np.inf
    hit10_cor = -np.inf
    epoch_list = []
    hit_list = []
    ndcg_list = []

    # train model
    for epoch in range(1, args.epoch_num + 1):
        time1 = time.time()
        model.train()
        total_loss = 0
        for idx, batch_data in enumerate(data_loader):
            batch_idx = idx + 1
            time2 = time.time()
            pad = 0
            src = batch_data['seq_in'].squeeze()
            src_dist = batch_data['dist_in'].squeeze()
            src_timediff = batch_data['timediff_in'].squeeze()
            src_mask = (src != pad).unsqueeze(-2)
            tgt = batch_data['target'].view(-1)
            if use_cuda:
                src = src.to(device)
                src_dist = src_dist.to(device)
                src_timediff = src_timediff.to(device)
                src_mask = src_mask.to(device)
                tgt = tgt.to(device)
            out = model.forward(src, src_dist, src_timediff, src_mask)
            batch_loss = model.loss(out, tgt)
            batch_loss.backward()
            cur_lr = optimizer.step()
            optimizer.optimizer.zero_grad()
            total_loss += batch_loss.item()
            if (batch_idx % args.print_every) == 0:
                logging.info('Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | L_Rate {:.5f} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, batch_idx, batch_num, time.time()-time2, cur_lr, batch_loss.item(), total_loss/batch_idx))
        logging.info('Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, batch_num, time.time()-time1, total_loss/batch_num))

        if (epoch % args.save_every) == 0:
            save_model(model, args.save_dir, epoch)
            last_save_epoch = epoch

        if (epoch % args.evaluate_every) == 0:
            time3 = time.time()
            val_data = NextPOIDataset(phase='val',
                                      poi_vocab=args.poi_vocab,
                                      data_root=args.data_root,
                                      poi_maxlen=args.poi_maxlen,
                                      logging=logging,
                                      test_num_neg=args.test_num_neg)
            hits, _, ndcgs = evaluate(model, val_data, args.test_batch_size, [5,10], use_cuda, device, args.test_num_neg)
            logging.info('Evaluation (K={}): Epoch {:04d} | Total Time {:.1f}s | Hit {:.4f} NDCG {:.4f}'.format(5, epoch, time.time() - time3, hits[0], ndcgs[0]))
            logging.info('Evaluation (K={}): Epoch {:04d} | Total Time {:.1f}s | Hit {:.4f} NDCG {:.4f}'.format(10, epoch, time.time() - time3, hits[1], ndcgs[1]))
            epoch_list.append(epoch)
            hit_list.append(hits)
            ndcg_list.append(ndcgs)

            # save the best result
            if ndcgs[0] > ndcg5_max:
                ndcg5_max, ndcg10_cor = ndcgs
                hit5_cor, hit10_cor = hits
                save_model(model, args.save_dir, epoch, best_epoch)
                best_epoch = epoch

            metrics = pd.DataFrame([epoch, hits, ndcgs]).transpose()
            metrics.to_csv(os.path.join(args.save_dir, 'train_results.csv'), mode='a', header=False, sep='\t',
                           index=False)

    best_metrics = pd.DataFrame([best_epoch, [hit5_cor, hit10_cor], [ndcg5_max, ndcg10_cor]]).transpose()
    best_metrics.to_csv(os.path.join(args.save_dir, 'train_results.csv'), mode='a', header=False, sep='\t', index=False)

    if last_save_epoch > -1:
        return os.path.join(args.save_dir, 'model_{}.pth'.format(last_save_epoch)), last_save_epoch + 1
    else: return None, None


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    train(args)