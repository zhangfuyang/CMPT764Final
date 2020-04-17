import os
from argparse import ArgumentParser

def get_args():
	parser = ArgumentParser(description='grass_pytorch')
	parser.add_argument('--boxSize', type=int, default=12)
	parser.add_argument('--featureSize', type=int, default=80)
	parser.add_argument('--hiddenSize', type=int, default=200)
	parser.add_argument('--symmetrySize', type=int, default=8)
	parser.add_argument('--vqDictionary', type=int, default=2048)
	parser.add_argument('--vqFeature', type=int, default=8)


	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=123)
	parser.add_argument('--show_log_every', type=int, default=3)
	parser.add_argument('--save_log', action='store_true', default=False)
	parser.add_argument('--save_log_every', type=int, default=3)
	parser.add_argument('--save_snapshot', action='store_true', default=False)
	parser.add_argument('--save_snapshot_every', type=int, default=5)
	parser.add_argument('--no_plot', action='store_true', default=False)
	parser.add_argument('--lr', type=float, default=.001)
	parser.add_argument('--lr_decay_by', type=float, default=1)
	parser.add_argument('--lr_decay_every', type=float, default=1)

	parser.add_argument('--no_cuda', action='store_true', default=False)
	parser.add_argument('--finetune', action='store_true', default=False)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--data_path', type=str, default='data')
	parser.add_argument('--save_path', type=str, default='models')
	parser.add_argument('--resume_snapshot', type=str, default='')
	parser.add_argument('--testset', type=str, default='B')#A:3 B:5 C:10
	parser.add_argument('--sample_shape_number', type=int, default=2)#dont change
	args = parser.parse_args()
	return args