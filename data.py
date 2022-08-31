'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of STAR-HiT model in:
Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation
'''
import os
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset


class NextPOIDataset(Dataset):
    def __init__(self, phase, poi_vocab=None, data_root=None, poi_maxlen=64, logging=None, test_num_neg=None):
        assert data_root is not None
        assert phase in ["train", "val", "test"], \
            "Phase must be one of train, val, test!"
        self.poi_maxlen = poi_maxlen
        self.phase = phase
        self.poi_vocab = poi_vocab
        self.poi_seq = []
        self.poi_tgt = []
        self.poi_dist_mat_in = []
        self.poi_timediff_mat_in = []
        self.test_num_neg = test_num_neg

        data_path = os.path.join(data_root, '{}.pkl'.format(phase))
        data_dict = joblib.load(data_path)
        data_idxes = np.arange(len(data_dict))

        for i in data_idxes:
            self.poi_seq.append(data_dict[i]['poi_id_seq_in'])
            self.poi_tgt.append(data_dict[i]['poi_out'])
            self.poi_dist_mat_in.append(data_dict[i]['poi_dist_mat_in'])
            self.poi_timediff_mat_in.append(data_dict[i]['poi_timediff_mat_in'])

        if self.test_num_neg:
            poi_tgt_test_file = os.path.join(data_root, 'test_{}.pkl'.format(self.test_num_neg))
            if not os.path.isfile(poi_tgt_test_file):
                self.poi_tgt_test_dict = {}
                self.poi_tgt_test = self.test_neg_sampling(self.poi_tgt, self.test_num_neg, poi_tgt_test_file)
            else:
                self.poi_tgt_test = joblib.load(poi_tgt_test_file)

        self.phase = phase
        self.data_path = data_path
        if logging:
            self.print_info(logging)

    def test_neg_sampling(self, poi_tgt, num_neg=5000, save_file=None):
        poi_tgt_tests = []
        for poi in poi_tgt:
            if poi in self.poi_tgt_test_dict.keys():
                poi_tgt_test = self.poi_tgt_test_dict[poi]
            else:
                poi_tgt_test = [poi]
                for _ in range(num_neg):
                    poi_neg = np.random.randint(1, self.poi_vocab)
                    while poi_neg in poi_tgt_test:
                        poi_neg = np.random.randint(1, self.poi_vocab)
                    poi_tgt_test.append(poi_neg)
                self.poi_tgt_test_dict[poi] = poi_tgt_test
            poi_tgt_tests.append(poi_tgt_test)
        if save_file:
            joblib.dump(poi_tgt_tests, save_file)
        return poi_tgt_tests

    def __len__(self):
        return len(self.poi_tgt)

    def __getitem__(self, index):
        seq_length = len(self.poi_seq[index])
        self.poi_seq_padded = np.zeros([self.poi_maxlen], dtype=np.int32)
        self.poi_seq_padded[:seq_length] = self.poi_seq[index]

        self.poi_dist_padded = np.zeros([self.poi_maxlen, self.poi_maxlen], dtype=np.float32)
        self.poi_dist_padded[:seq_length,:seq_length] = self.poi_dist_mat_in[index]

        self.poi_timediff_padded = np.zeros([self.poi_maxlen, self.poi_maxlen], dtype=np.float32)
        self.poi_timediff_padded[:seq_length,:seq_length] = self.poi_timediff_mat_in[index]

        if self.phase is not 'train' and self.test_num_neg:
            tgt = torch.tensor(self.poi_tgt_test[index]).long()
        else:
            tgt = torch.tensor([self.poi_tgt[index]]).long()

        samples = {
            'seq_in': torch.tensor([self.poi_seq_padded]).long(),
            'dist_in': torch.tensor([self.poi_dist_padded]),
            'timediff_in': torch.tensor([self.poi_timediff_padded]),
            'target': tgt
        }
        return samples

    def print_info(self, logging):
        logging.info('current phase: {}'.format(self.phase))
        logging.info('current data path: {}'.format(self.data_path))
        logging.info('the number of samples: {}'.format(len(self.poi_tgt)))
        logging.info('the max length of the sequence: {}'.format(self.poi_maxlen))


if __name__ == "__main__":
    ### test only
    pass


