'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of STAR-HiT model in:
Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math


def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SubSeqCoder(nn.Module):
	def __init__(self, input_len, new_seq_len, weights=(1., 1.), sample_num=2, width_bias=None):
		super(SubSeqCoder, self).__init__()
		self.input_len = input_len  # the length of the input sequence
		self.new_seq_len = new_seq_len  # the count of subsequences of the sequence
		self.weights = weights  # 2d: center coordinate and length
		self.sample_num = sample_num  # the number of sampling points in each subsequence
		self._generate_anchor()  # return the anchor x of each subsequence
		self.width_bias = None
		if width_bias is not None:
			self.width_bias = nn.Parameter(width_bias)

	def _generate_anchor(self):
		anchors = []
		subseq_stride = 1. / self.new_seq_len
		for i in range(self.new_seq_len):
			anchors.append((0.5 + i) * subseq_stride)
		anchors = torch.as_tensor(anchors)
		self.register_buffer("anchor", anchors)

	def forward(self, pred_offset):
		if self.width_bias is not None:
			pred_offset[:, :, -1] = pred_offset[:, :, -1] + self.width_bias
		boxes = self.decode(pred_offset)
		points = self.sample(boxes)
		return boxes, points

	def decode(self, rel_codes):
		boxes = self.anchor
		poi = 1. / self.new_seq_len
		w_x, w_width = self.weights

		dx = torch.tanh(rel_codes[:, :, 0] / w_x) * poi
		dw = F.relu(torch.tanh(rel_codes[:, :, -1] / w_width)) * poi

		pred_boxes = torch.zeros_like(rel_codes)
		ref_x = boxes.unsqueeze(0)
		pred_boxes[:, :, 0] = ref_x + dx - dw
		pred_boxes[:, :, -1] = ref_x + dx + dw
		pred_boxes = pred_boxes.clamp_(min=0., max=1.)
		return pred_boxes

	def sample(self, boxes_x):
		boxes_x = F.interpolate(boxes_x, size=self.sample_num, mode='linear', align_corners=True)
		boxes_y = torch.zeros_like(boxes_x)
		boxes = torch.stack([boxes_x, boxes_y], dim=-1)
		return boxes


class SequencePartition(nn.Module):
	def __init__(self, seq_length, sub_seq_len, new_seq_len, sample_num, emb_size):
		super(SequencePartition, self).__init__()
		self.total_sample_num = new_seq_len * sample_num
		self.activation = nn.ReLU()
		if seq_length > 1:
			subseq_stride = math.ceil(seq_length / new_seq_len)
			subseq_padding = math.ceil((subseq_stride * (new_seq_len - 1) + sub_seq_len - seq_length) / 2)
			self.proj = nn.Conv1d(emb_size, emb_size, kernel_size=sub_seq_len, stride=subseq_stride,
			                      padding=subseq_padding)
		else:
			self.proj = nn.Conv1d(emb_size, emb_size, kernel_size=1)

		self.offset_activation = nn.ReLU()
		self.offset_predictor = nn.Linear(emb_size, 2)
		self.subseq_coder = SubSeqCoder(input_len=seq_length, new_seq_len=new_seq_len, weights=(1., 1.),
		                                sample_num=sample_num, width_bias=torch.tensor(5./3.).sqrt().log())
		self.sampling_boxes = None

	def forward(self, x):
		src = x
		x = x.permute(0, 2, 1)
		x = self.proj(x)
		x = x.permute(0, 2, 1)
		pred_offset = self.offset_predictor(self.offset_activation(x))
		sampling_boxes, sampling_loc = self.subseq_coder(pred_offset)
		sampling_point = self.sample(src, sampling_loc)
		self.sampling_boxes = sampling_boxes
		return sampling_point.permute(0, 2, 3, 1)

	def sample(self, src, sampling_loc):
		sampling_grid = 2 * sampling_loc - 1
		src = src.transpose(1, 2).unsqueeze(-2)
		sampling_value = F.grid_sample(src, sampling_grid, mode='nearest', padding_mode='border')
		return sampling_value

	def reset_offset(self):
		nn.init.constant_(self.offset_predictor.weight, 0)
		if hasattr(self.offset_predictor, "bias") and self.offset_predictor.bias is not None:
			nn.init.constant_(self.offset_predictor.bias, 0)
		print("Parameter of offsets reset.")


class SubsequenceAggregation(nn.Module):
	def __init__(self, emb_size, sub_seq_len, new_seq_len, hid_size=None, sublayer_drop_p=0.1):
		super(SubsequenceAggregation, self).__init__()
		self.emb_size = emb_size
		self.sub_seq_len = sub_seq_len
		self.new_seq_len = new_seq_len
		self.pooling = nn.AvgPool1d(kernel_size=sub_seq_len, stride=sub_seq_len)
		hid_size = hid_size or emb_size * 2
		self.ffn = FeedForward(emb_size, hid_size)
		self.sublayer = SublayerConnection(emb_size, sublayer_drop_p)

	def forward(self, x):
		# input: B * NewSeqLen * SubSeqLen * C
		x = x.permute(0, 3, 1, 2).contiguous().view(-1, self.emb_size, self.new_seq_len * self.sub_seq_len)
		# output: B * C * (NewSeqLen * SubSeqLen)
		x = self.pooling(x).transpose(1, 2)
		# output: B * NewSeqLen * C
		return self.sublayer(x, self.ffn)


class FeedForward(nn.Module):
	def __init__(self, in_size, hid_size=None, out_size=None, activation=nn.ReLU, drop_p=0.1):
		super(FeedForward, self).__init__()
		out_size = out_size or in_size
		hid_size = hid_size or in_size
		self.ffn = nn.Sequential(
			nn.Linear(in_size, hid_size),
			activation(inplace=True),
			nn.Dropout(drop_p),
			nn.Linear(hid_size, out_size),
			nn.Dropout(drop_p))

	def forward(self, x):
		x = self.ffn(x)
		return x


class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		return x + self.dropout(sublayer(self.norm(x)))


class PositionalEncoding(nn.Module):
	def __init__(self, emb_size, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, emb_size)
		position = torch.arange(0., max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0., emb_size, 2) *
		                     -(math.log(10000.0) / emb_size))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:, :x.size(1)]
		return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
	"""
    attention(q,k,v) = softmax(q k^T / sqrt(d_k)) v
    """
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class LocalAttention(nn.Module):
	def __init__(self, emb_size, attn_drop_p=0.1):
		super(LocalAttention, self).__init__()
		self.linears = clones(nn.Linear(emb_size, emb_size), 4)
		self.attn = None
		self.attn_dropout = nn.Dropout(p=attn_drop_p)

	def forward(self, query, key, value):
		query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
		x, self.attn = attention(query, key, value, dropout=self.attn_dropout)
		return self.linears[-1](x)


class LocalAttentionLayer(nn.Module):
	def __init__(self, emb_size, sub_seq_len, new_seq_len, hid_size=None, attn_drop_p=0.1, sublayer_drop_p=0.1):
		super(LocalAttentionLayer, self).__init__()
		self.emb_size = emb_size
		self.sub_seq_len = sub_seq_len
		self.new_seq_len = new_seq_len
		self.local_attn = LocalAttention(emb_size, attn_drop_p)
		hid_size = hid_size or emb_size * 2
		self.local_ffn = FeedForward(emb_size, hid_size)
		self.sublayers = clones(SublayerConnection(emb_size, sublayer_drop_p), 2)

	def forward(self, x):
		x = self.sublayers[0](x, lambda x: self.local_attn(x, x, x))
		x = self.sublayers[1](x, self.local_ffn)
		return x


class GlobalAttention(nn.Module):
	def __init__(self, emb_size, head_num=4, attn_drop_p=0.1):
		super(GlobalAttention, self).__init__()
		assert emb_size % head_num == 0
		self.d_k = emb_size // head_num
		self.head_num = head_num
		self.linears = clones(nn.Linear(emb_size, emb_size), 4)
		self.attn = None
		self.attn_dropout = nn.Dropout(p=attn_drop_p)

	def forward(self, query, key, value, mask=None):
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		batch_size = query.size(0)
		# 1) Do all the linear projections in batch from d to h x d_k.
		query, key, value = [l(x).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2) for l, x in
		                     zip(self.linears, (query, key, value))]
		# 2) Apply attention on all the projected vectors in batch.
		x, self.attn = attention(query, key, value, mask=mask, dropout=self.attn_dropout)
		# 3) Concatenate using a view and apply a final linear.
		x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)
		return self.linears[-1](x)


class GlobalAttentionLayer(nn.Module):
	def __init__(self, emb_size, head_num, hid_size=None, attn_drop_p=0.1, sublayer_drop_p=0.1):
		super(GlobalAttentionLayer, self).__init__()
		self.global_attn = GlobalAttention(emb_size, head_num, attn_drop_p)
		hid_size = hid_size or emb_size * 2
		self.global_ffn = FeedForward(emb_size, hid_size)
		self.sublayers = clones(SublayerConnection(emb_size, sublayer_drop_p), 2)

	def forward(self, x, mask):
		x = self.sublayers[0](x, lambda x: self.global_attn(x, x, x, mask))
		x = self.sublayers[1](x, self.global_ffn)
		return x


class Block(nn.Module):
	def __init__(self, seq_len, sub_seq_len, new_seq_len, sample_num, emb_size, hid_size, head_num,
	             global_attn_drop_p=0.1, local_attn_drop_p=0.1):
		super(Block, self).__init__()
		self.sub_seq_len = sub_seq_len
		self.new_seq_len = new_seq_len
		self.global_attn = GlobalAttentionLayer(emb_size, head_num, hid_size, global_attn_drop_p)
		self.local_attn = LocalAttentionLayer(emb_size, sub_seq_len, new_seq_len, hid_size, local_attn_drop_p)
		self.partition = SequencePartition(seq_len, sub_seq_len, new_seq_len, sample_num, emb_size)
		self.partition.reset_offset()
		self.aggragation = SubsequenceAggregation(emb_size, sub_seq_len, new_seq_len, hid_size)

	def forward(self, x, mask=None):
		x = self.global_attn(x, mask)  # output: B * SeqLen * C
		x = self.partition(x)  # output: B * NewSeqLen * SubSeqLen * C
		x = self.local_attn(x)  # output: B * NewSeqLen * SubSeqLen * C
		x = self.aggragation(x) # output: B * NewSeqLen * C
		return x


class EmbeddingModule(nn.Module):
	def __init__(self, poi_vocab, poi_maxlen, emb_size, pos_drop_p=0.1):
		super(EmbeddingModule, self).__init__()
		self.emb_size = emb_size
		self.poi_emb = nn.Embedding(poi_vocab, emb_size)
		self.linear = nn.Linear(emb_size + poi_maxlen, emb_size)
		self.lambd_time = nn.Parameter(torch.ones(poi_maxlen))
		self.k_dist = nn.Parameter(torch.ones(poi_maxlen))
		self.pos_emb = PositionalEncoding(emb_size, pos_drop_p)

	def forward(self, src, src_s, src_t, eps=1e-9):
		src_emb = self.poi_emb(src) * math.sqrt(self.emb_size)
		src_s += eps
		src_t += eps
		st_mat = self.lambd_time * src_t.log() + self.k_dist * src_s.log()
		x = torch.cat((src_emb, st_mat), -1)
		x = self.linear(x)
		return self.pos_emb(x)


class Predictor(nn.Module):
	def __init__(self, emb_size, poi_vocab):
		super(Predictor, self).__init__()
		self.proj = nn.Linear(emb_size, poi_vocab)

	def forward(self, x):
		x = torch.sum(x, dim=-2)
		return F.log_softmax(self.proj(x), dim=-1)


class STARHiT(nn.Module):
	def __init__(self, poi_vocab, poi_maxlen, emb_size=64, hid_size=128, head_num=4, block_num=2, sub_seq_len=8, sample_num=None, dropout=0.1):
		super(STARHiT, self).__init__()
		self.emb = EmbeddingModule(poi_vocab, poi_maxlen, emb_size, pos_drop_p=dropout)
		seq_len_in = [math.ceil(poi_maxlen / (sub_seq_len ** i)) for i in range(block_num)]
		seq_len_out = [math.ceil(poi_maxlen / (sub_seq_len ** i)) for i in range(1, block_num + 1)]
		sample_num = sample_num or sub_seq_len
		self.blocks = nn.ModuleList(
			[Block(seq_len_in[i], sub_seq_len, seq_len_out[i], sample_num, emb_size, hid_size, head_num) for i
			 in range(block_num)])
		self.predictor = Predictor(emb_size, poi_vocab)
		self.loss = nn.NLLLoss()  # already log_softmax

	def forward(self, src, src_dist, src_timediff, src_mask=None):
		src_emb = self.emb(src, src_dist, src_timediff)
		src_emb = self.blocks[0](src_emb, src_mask)
		for block in self.blocks[1:]:
			src_emb = block(src_emb)
		return self.predictor(src_emb)


if __name__ == "__main__":
	# for test only
	pass