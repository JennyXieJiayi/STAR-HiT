'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of STAR-HiT model in:
Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation
'''
import torch
import numpy as np
from torch.utils.data import DataLoader


def evaluate(trained_model, data, batch_size, K=[5,10], use_cuda=True, device='cuda', num_neg=None):
    trained_model.eval()
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    total_sample_num = len(data)
    global_hit_at_k = np.array([0] * len(K))
    global_ndcg_at_k = np.array([0] * len(K))
    global_mrr = np.array([0] * len(K))

    with torch.no_grad():
        for idx, batch_data in enumerate(data_loader):
            pad = 0
            src = batch_data['seq_in'].squeeze()
            src_dist = batch_data['dist_in'].squeeze()
            src_timediff = batch_data['timediff_in'].squeeze()
            src_mask = (src != pad).unsqueeze(-2)
            if use_cuda:
                trained_model.to(device)
                src = src.to(device)
                src_dist = src_dist.to(device)
                src_timediff = src_timediff.to(device)
                src_mask = src_mask.to(device)
            out_prob = trained_model.forward(src, src_dist, src_timediff, src_mask) # the prob
            if num_neg == None:
                tgt = batch_data['target']
            else:
                out_prob = out_prob.gather(1, batch_data['target'].to(device))
                tgt = torch.zeros((out_prob.shape[0], 1)).long()

            _, out_rank = torch.sort(out_prob, descending=True)
            if use_cuda:
                out_rank = out_rank.cpu()

            for idx_k, k in enumerate(K):
                global_hit_at_k[idx_k] += hit_at_k_per_batch(out_rank, tgt, k)
                global_ndcg_at_k[idx_k] += ndcg_at_k_per_batch(out_rank, tgt, k)
                if k == 20:
                    global_mrr[idx_k] += mrr_per_batch(out_rank, tgt)

    global_hit_at_k = global_hit_at_k / total_sample_num
    global_ndcg_at_k = global_ndcg_at_k / total_sample_num
    global_mrr = global_mrr / total_sample_num
    return global_hit_at_k, global_mrr, global_ndcg_at_k


def hit_at_k_per_batch(pred, tgt, k):
    hits_num = 0
    for i in range(len(tgt)):
        tgt_set = set(tgt[i].numpy())
        pred_set = set(pred[i][:k].numpy())
        hits_num += len(tgt_set & pred_set)
    return hits_num


def recall_at_k_per_batch(pred, tgt, k):
    sum_recall = 0.
    num_sample = 0
    for i in range(len(tgt)):
        tgt_set = set(tgt[i].numpy())
        pred_set = set(pred[i][:k].numpy())
        if len(tgt_set) != 0:
            sum_recall += len(tgt_set & pred_set) / float(len(tgt_set))
            num_sample += 1
    return num_sample, sum_recall


def precision_at_k_per_batch(pred, tgt, k):
    sum_precision = 0.
    num_sample = 0
    for i in range(len(tgt)):
        tgt_set = set(tgt[i].numpy())
        pred_set = set(pred[i][:k].numpy())
        if len(tgt_set) != 0:
            sum_precision += len(tgt_set & pred_set) / float(k)
            num_sample += 1
    return num_sample, sum_precision


def mrr_per_batch(pred, tgt):
    score = 0.
    for i in range(len(tgt)):
        sample_pred = pred[i]
        sample_tgt = tgt[i]
        sample_score = 0.0
        num_tgt = len(tgt[i])
        for j, p in enumerate(sample_pred):
            if p in sample_tgt and p not in sample_pred[:j]:
                sample_score += 1 / (j + 1.0)
                num_tgt -= 1
                if num_tgt <= 0:
                    break
        score += sample_score / len(sample_tgt)
    return score


def mrr_at_k_per_batch(pred, tgt, k):
    score = 0.
    for i in range(len(tgt)):
        sample_pred = pred[i,:k]
        sample_tgt = tgt[i]
        sample_score = 0.0
        for j, p in enumerate(sample_pred):
            if p in sample_tgt and p not in sample_pred[:j]:
                sample_score += 1 / (j + 1.0)
        score += sample_score / min(len(sample_tgt), k)
    return score


def ndcg_at_k_per_batch(pred, tgt, k):
    ndcg_score = 0.
    for i in range(len(tgt)):
        sample_pred = pred[i, :k].numpy()
        sample_tgt = tgt[i].numpy()
        ndcg_score += ndcg_at_k_per_sample(sample_pred, sample_tgt)
    return ndcg_score


def ndcg_at_k_per_sample(pred, tgt, method=1):
    r = np.zeros_like(pred, dtype=np.float32)
    ideal_r = np.zeros_like(pred, dtype=np.float32)
    for i, v in enumerate(pred):
        if v in tgt and v not in pred[:i]:
            r[i] = 1.
    ideal_r[:len(tgt)] = 1.

    idcg = dcg_at_k_per_sample(ideal_r, method)
    if not idcg:
        return 0.
    return dcg_at_k_per_sample(r, method) / idcg


def dcg_at_k_per_sample(r, method=1):
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


if __name__ == "__main__":
    # for test only
    pass