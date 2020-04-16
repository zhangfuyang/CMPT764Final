import torch
from torch import nn
from torch.autograd import Variable
from modelVQContext import GRASSDecoder
from draw3dOBB import showGenshape
from chairDataset import ChairDataset
import util
from torchfoldext import FoldExt
from modelVQContext import GRASSMerge
from chairDataset import Tree
import math
import random
import numpy as np
from scipy.io import savemat

def vrrotvec2mat(rotvector):
	s = math.sin(rotvector[3])
	c = math.cos(rotvector[3])
	t = 1 - c
	x = rotvector[0]
	y = rotvector[1]
	z = rotvector[2]
	m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]).cuda()
	return m

def decode_structure(root, noise=False):
	"""
	Decode a root code into a tree structure of boxes
	"""
	# decode = model.sampleDecoder(root_code)
	syms = [torch.ones(8).mul(10)]
	stack = [root]
	boxes = []
	labels = []
	while len(stack) > 0:
		node = stack.pop()
		# label_prob = model.nodeClassifier(f)
		# _, label = torch.max(label_prob, 1)
		#label = node.label.item()
		if node.is_adj():  # ADJ
			# left, right = model.adjDecoder(f)
			stack.append(node.left)
			stack.append(node.right)
			s = syms.pop()
			syms.append(s)
			syms.append(s)
		if node.is_sym():  # SYM
			# left, s = model.symDecoder(f)
			# s = s.squeeze(0)
			stack.append(node.left)
			syms.pop()
			if noise:
				syms.append(node.sym_noise.squeeze(0))
			else:
				syms.append(node.sym.squeeze(0))
		if node.is_leaf():	# BOX
			if noise:
				reBox = node.box_noise
			else:
				reBox = node.box
			reBoxes = [reBox]
			reLabels = [node.box_label]
			s = syms.pop()
			l1 = abs(s[0] + 1)
			l2 = abs(s[0])
			l3 = abs(s[0] - 1)
			if l1 < 0.15:
				sList = torch.split(s, 1, 0)
				bList = torch.split(reBox.data.squeeze(0), 1, 0)
				f1 = torch.cat([sList[1], sList[2], sList[3]])
				f1 = f1/torch.norm(f1)
				f2 = torch.cat([sList[4], sList[5], sList[6]])
				folds = round(1/s[7].item())
				for i in range(folds-1):
					rotvector = torch.cat([f1, sList[7].mul(2*3.1415).mul(i+1)])
					rotm = vrrotvec2mat(rotvector)
					center = torch.cat([bList[0], bList[1], bList[2]])
					dir0 = torch.cat([bList[3], bList[4], bList[5]])
					dir1 = torch.cat([bList[6], bList[7], bList[8]])
					dir2 = torch.cat([bList[9], bList[10], bList[11]])
					newcenter = rotm.matmul(center.add(-f2)).add(f2)
					newdir1 = rotm.matmul(dir1)
					newdir2 = rotm.matmul(dir2)
					newbox = torch.cat([newcenter, dir0, newdir1, newdir2])
					reBoxes.append(newbox.unsqueeze(0))
					reLabels.append(node.box_label)
			if l3 < 0.15:
				sList = torch.split(s, 1, 0)
				bList = torch.split(reBox.data.squeeze(0), 1, 0)
				trans = torch.cat([sList[1], sList[2], sList[3]])
				trans_end = torch.cat([sList[4], sList[5], sList[6]])
				center = torch.cat([bList[0], bList[1], bList[2]])
				trans_length = math.sqrt(torch.sum(trans**2))
				trans_total = math.sqrt(torch.sum(trans_end.add(-center)**2))
				folds = round(trans_total/trans_length)
				for i in range(folds):
					center = torch.cat([bList[0], bList[1], bList[2]])
					dir0 = torch.cat([bList[3], bList[4], bList[5]])
					dir1 = torch.cat([bList[6], bList[7], bList[8]])
					dir2 = torch.cat([bList[9], bList[10], bList[11]])
					newcenter = center.add(trans.mul(i+1))
					newbox = torch.cat([newcenter, dir0, dir1, dir2])
					reBoxes.append(newbox.unsqueeze(0))
					reLabels.append(node.box_label)
			if l2 < 0.15:
				sList = torch.split(s, 1, 0)
				bList = torch.split(reBox.data.squeeze(0), 1, 0)
				ref_normal = torch.cat([sList[1], sList[2], sList[3]])
				ref_normal = ref_normal/torch.norm(ref_normal)
				ref_point = torch.cat([sList[4], sList[5], sList[6]])
				center = torch.cat([bList[0], bList[1], bList[2]])
				dir0 = torch.cat([bList[3], bList[4], bList[5]])
				dir1 = torch.cat([bList[6], bList[7], bList[8]])
				dir2 = torch.cat([bList[9], bList[10], bList[11]])
				if ref_normal.matmul(ref_point.add(-center)) < 0:
					ref_normal = -ref_normal
				newcenter = ref_normal.mul(2*abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
				if ref_normal.matmul(dir1) < 0:
					ref_normal = -ref_normal
				dir1 = dir1.add(ref_normal.mul(-2*ref_normal.matmul(dir1)))
				if ref_normal.matmul(dir2) < 0:
					ref_normal = -ref_normal
				dir2 = dir2.add(ref_normal.mul(-2*ref_normal.matmul(dir2)))
				newbox = torch.cat([newcenter, dir0, dir1, dir2])
				reBoxes.append(newbox.unsqueeze(0))
				reLabels.append(node.box_label)

			boxes.extend(reBoxes)
			labels.extend(reLabels)
	return boxes, labels

def render_node_to_box(tree):
	Boxs = []
	syms = [torch.ones(8).mul(10)]
	stack = [tree.root]
	while len(stack) > 0:
		node = stack.pop()
		if node.is_adj():
			s = syms.pop()
			syms.append(s)
			syms.append(s)
			stack.append(node.left)
			stack.append(node.right)
		elif node.is_sym():
			stack.append(node.left)
			syms.pop()
			syms.append(node.sym.squeeze(0))
		elif node.is_leaf():
			reBoxes = [node.box]
			s = syms.pop()
			l1 = abs(s[0]+1)
			l2 = abs(s[0])
			l3 = abs(s[0]-1)

			if l1 < 0.15:
				sList = torch.split(s, 1, 0)
				bList = torch.split(node.box.squeeze(0), 1, 0)
				f1 = torch.cat([sList[1], sList[2], sList[3]])
				f1 = f1/torch.norm(f1)
				f2 = torch.cat([sList[4], sList[5], sList[6]])
				folds = round(1/s[7])
				for i in range(folds-1):
					rotvector = torch.cat([f1, sList[7].mul(2*3.1415).mul(i+1)])
					rotm = vrrotvec2mat(rotvector)
					center = torch.cat([bList[0], bList[1], bList[2]])
					dir0 = torch.cat([bList[3], bList[4], bList[5]])
					dir1 = torch.cat([bList[6], bList[7], bList[8]])
					dir2 = torch.cat([bList[9], bList[10], bList[11]])
					newcenter = rotm.matmul(center.add(-f2)).add(f2)
					newdir1 = rotm.matmul(dir1)
					newdir2 = rotm.matmul(dir2)
					newbox = torch.cat([newcenter, dir0, newdir1, newdir2])
					reBoxes.append(Variable(newbox.unsqueeze(0)))
			elif l2 < 0.15:
				sList = torch.split(s, 1, 0)
				bList = torch.split(node.box.squeeze(0), 1, 0)
				trans = torch.cat([sList[1], sList[2], sList[3]])
				trans_end = torch.cat([sList[4], sList[5], sList[6]])
				center = torch.cat([bList[0], bList[1], bList[2]])
				trans_length = math.sqrt(torch.sum(trans**2))
				trans_total = math.sqrt(torch.sum(trans_end.add(-center)**2))
				folds = round(trans_total/trans_length)
				for i in range(folds):
					center = torch.cat([bList[0], bList[1], bList[2]])
					dir0 = torch.cat([bList[3], bList[4], bList[5]])
					dir1 = torch.cat([bList[6], bList[7], bList[8]])
					dir2 = torch.cat([bList[9], bList[10], bList[11]])
					newcenter = center.add(trans.mul(i+1))
					newbox = torch.cat([newcenter, dir0, dir1, dir2])
					reBoxes.append(Variable(newbox.unsqueeze(0)))
			elif l3 < 0.15:
				sList = torch.split(s, 1, 0)
				bList = torch.split(node.box.squeeze(0), 1, 0)
				ref_normal = torch.cat([sList[1], sList[2], sList[3]])
				ref_normal = ref_normal / torch.norm(ref_normal)
				ref_point = torch.cat([sList[4], sList[5], sList[6]])
				center = torch.cat([bList[0], bList[1], bList[2]])
				dir0 = torch.cat([bList[3], bList[4], bList[5]])
				dir1 = torch.cat([bList[6], bList[7], bList[8]])
				dir2 = torch.cat([bList[9], bList[10], bList[11]])
				if ref_normal.matmul(ref_point.add(-center)) < 0:
					ref_normal = -ref_normal
				newcenter = ref_normal.mul(2 * abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
				if ref_normal.matmul(dir1) < 0:
					ref_normal = -ref_normal
				dir1 = dir1.add(ref_normal.mul(-2 * ref_normal.matmul(dir1)))
				if ref_normal.matmul(dir2) < 0:
					ref_normal = -ref_normal
				dir2 = dir2.add(ref_normal.mul(-2 * ref_normal.matmul(dir2)))
				newbox = torch.cat([newcenter, dir0, dir1, dir2])
				reBoxes.append(Variable(newbox.unsqueeze(0)))

			Boxs.extend(reBoxes)
	return Boxs

def encode_tree(model, tree):
	def encode_node(node):
		if node.is_leaf():
			if config.finetune:
				n = model.leafNode(node.box_noise)
			else:
				n = model.leafNode(node.box)
			return n
		if node.is_adj():
			left = encode_node(node.left)
			right = encode_node(node.right)
			n = model.adjNode(left, right)
			return n
		if node.is_sym():
			feature = encode_node(node.left)
			if config.finetune:
				n = model.symNode(feature, node.sym_noise)
			else:
				n = model.symNode(feature, node.sym)
			return n

	encoding = encode_node(tree.root)
	return encoding

def decode_tree(model, feature, tree):
	def encode_node(node):
		if node.is_leaf():
			if config.finetune:
				b = node.box_noise
			else:
				b = node.box
			if config.cuda:
				n = model.leafNode(b.cuda())
			else:
				n = model.leafNode(b)
			return n
		if node.is_adj():
			left = encode_node(node.left)
			right = encode_node(node.right)
			n = model.adjNode(left, right)
			return n
		if node.is_sym():
			feature = encode_node(node.left)
			if config.finetune:
				s = node.sym_noise
			else:
				s = node.sym
			if config.cuda:
				n = model.symNode(feature, s.cuda())
			else:
				n = model.symNode(feature, s)
			return n

	def decode_node(feature, node):
		if node.is_leaf():
			return 0
		elif node.is_adj():
			node_loss, feature = model.vqlizationWithLoss(feature)
			left_node = node.left
			right_node = node.right
			fl = encode_node(left_node)
			fr = encode_node(right_node)
			left_f = model.outterNode(feature, fr)
			right_f = model.outterNode(feature, fl)
			left_loss = decode_node(left_f, node.left)
			right_loss = decode_node(right_f, node.right)
			return left_loss + right_loss + node_loss
		elif node.is_sym():
			node_loss, feature = model.vqlizationWithLoss(feature)
			f = encode_node(node)
			new_f, sym_f = model.symParaNode(feature, f)
			left_loss = decode_node(new_f, node.left)
			return left_loss + node_loss

	loss = decode_node(feature, tree.root)
	return loss


def my_collate(batch):
	return batch


def inference(example):
	enc_fold = FoldExt(cuda=config.cuda)
	enc_fold_nodes = []
	enc_fold_nodes.append(encode_fold(enc_fold, example))
	enc_fold_nodes = enc_fold.apply(model, [enc_fold_nodes])
	enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)
	refineboxes = []
	syms = []
	Labels = []
	decode_fold(model, enc_fold_nodes[0], example, refineboxes, syms, Labels)
	refineboxes = torch.cat(refineboxes, 0)
	refineLabels = torch.Tensor(Labels, 0)
	if len(syms) == 0:
		syms = torch.zeros((1, 8))
	else:
		syms = torch.cat(syms, 0)
	refine_tree = Tree(refineboxes, example.ops, syms, refineLabels)
	return refine_tree


def dfs_b_find_min_id(tree_b):
	def dfs_id(node):
		global min_id
		global min_loss
		if node.is_leaf():
			if node.loss < min_loss:
				min_loss = node.loss
				min_id = node.id
		elif node.is_adj():
			dfs_id(node.left)
			dfs_id(node.right)
			if node.loss < min_loss:
				min_loss = node.loss
				min_id = node.id
		else:
			if node.loss < min_loss:
				min_loss = node.loss
				min_id = node.id
	global min_id 
	min_id = 0
	global min_loss 
	min_loss = 999
	dfs_id(tree_b.root)
	return min_id
		
def find_correspondence_loss(node_a, tree_b, model):
	def dfs_b(node):
		if node.is_adj():
			#check left child
			#print('node_a label, ', node_a.box_label)
			#print('node.left label, ', node.left.box_label)
			if node.left.box_label == node_a.box_label:
				print('replace node id of b: ', node.left.id)
				temp = node.left
				node.left = node_a
				#get error
				root_feature = encode_tree(model, tree_b)
				loss = decode_tree(model, root_feature, tree_b)
				#change to original
				node.left = temp
				node.left.loss = loss
				print('replace loss : ', loss)
			if node.left.is_adj():
				dfs_b(node.left) 
			#print('node_a label, ', node_a.box_label)
			#print('node.right label, ', node.right.box_label)
			if node.right.box_label == node_a.box_label:
				print('replace node id of b: ', node.right.id)
				temp = node.right
				node.right = node_a
				#get error
				root_feature = encode_tree(model, tree_b)
				loss = decode_tree(model, root_feature, tree_b)
				node.right = temp
				node.right.loss = loss
				print('replace loss : ', loss)
			if node.right.is_adj():
				dfs_b(node.right)
	dfs_b(tree_b.root)

def clean_tree_loss(node):
	if node.is_leaf():
		#assign loss to each node of tree b
		node.loss = 999
	elif node.is_adj():
		clean_tree_loss(node.left)
		clean_tree_loss(node.right)
		node.loss = 999
	else:
		node.loss = 999
		
def dfs_a(node, tree_b, model):
	#find which node of tree A need to be dealt with
	if node.is_leaf():
		print('check id of a leaf: ', node.id)
		#assign loss to each node of tree b
		find_correspondence_loss(node, tree_b, model)
		#find the best match node in tree b
		node.match_id = dfs_b_find_min_id(tree_b)
		print('find match id of b: ', node.match_id)
		clean_tree_loss(tree_b.root)
	elif node.is_adj():
		dfs_a(node.left, tree_b, model)
		dfs_a(node.right, tree_b, model)
	else:
		print('check id of a sym: ', node.id)
		#assign loss to each node of tree b
		find_correspondence_loss(node, tree_b, model)
		#find the best node in tree b
		node.match_id = dfs_b_find_min_id(tree_b)
		print('find match id of b: ', node.match_id)
		clean_tree_loss(tree_b.root)
   
def dfs_assign_label(node):
	if node.is_leaf():
		return node.box_label
	elif node.is_adj():
		left_label = dfs_assign_label(node.left)
		right_label = dfs_assign_label(node.right)
		if left_label == right_label:
			node.box_label = left_label
		return node.box_label
	else:
		left_label = dfs_assign_label(node.left)
		node.box_label = left_label
		return node.box_label

def find_box_from_node(node):
	if node.is_leaf():
		return node.box
	elif node.is_adj():
		left_boxes = find_box_from_node(node.left)
		right_boxes = find_box_from_node(node.right)
		return torch.cat((left_boxes, right_boxes), 0)
	else:
		left_boxes = find_box_from_node(node.left)
		return left_boxes
			
def find_box_from_tree_b(node, match_id):
	if node.is_leaf():
		if node.id == match_id:
			return node.box
	elif node.is_adj():
		if node.id == match_id:
			return find_box_from_node(node)
		#if children match
		left_result = find_box_from_tree_b(node.left, match_id)
		right_result = find_box_from_tree_b(node.right, match_id)
		if left_result is not None:
			return left_result
		if right_result is not None:
			return right_result
	else:
		if node.id == match_id:
			return find_box_from_node(node)
		
def show_correspondence(tree_a, tree_b):
	def dfs_a_show(node):
		if node.is_leaf():
			print('print node id of a, ', node.id)
			print('print node.match_id of b, ', node.match_id)
			if node.match_id == 0:
				print('No match found for this node!!! ')
				return
			box_b = find_box_from_tree_b(tree_b.root, node.match_id)
			#print('node.box, ', node.box.size())
			#print('box_b, ', box_b.size())
			boxes = torch.cat((node.box, box_b), 0)
			label_text = []
			label_text.append('shape_a_part:%d' % node.id)
			for i in range(box_b.size(0)):
				label_text.append('shape_b_part:%d' % node.match_id)
			showGenshape(boxes.data.cpu().numpy(), labels=label_text)
			return
		elif node.is_adj():
			dfs_a_show(node.left)
			dfs_a_show(node.right)
		else:
			print('print node id of a, ', node.id)
			print('print node.match_id of b, ', node.match_id)
			if node.match_id == 0:
				print('No match found for this node!!! ')
				return
			box_a = find_box_from_node(node)
			box_b = find_box_from_tree_b(tree_b.root, node.match_id)
			boxes = torch.cat((box_a, box_b), 0)
			label_text = []
			for i in range(box_a.size(0)):
				label_text.append('shape_a_part:%d' % node.id)
			label_text.append('shape_a_part:%d' % node.id)
			for i in range(box_b.size(0)):
				label_text.append('shape_b_part:%d' % node.match_id)
			showGenshape(boxes.data.cpu().numpy(), labels=label_text)
			return
	dfs_a_show(tree_a.root)

def give_valid_to_tree_b(tree_b, match_b_ids):
	def dfs_b_sample(node):
		if node.is_leaf():
			#not in match id
			#print('b, ', node.id)
			if node.id not in match_b_ids:
				#print('select_id b, ', node.id)
				node.selected = True
				return True
			else:
				return False
		elif node.is_adj():
			#print('b, ', node.id)
			if node.id == 0 or node.id not in match_b_ids:
				left_s = dfs_b_sample(node.left)
				right_s = dfs_b_sample(node.right)
				if left_s and right_s:
					#print('select_id b, ', node.id)
					node.selected = True
					return True
				else:
					node.selected = False
					return False
			else:
				node.selected = False
				return False
		else:
			#print('b, ', node.id)
			if node.id not in match_b_ids:
				#print('select_id b, ', node.id)
				node.selected = True
				return True
			else:
				return False
				
	dfs_b_sample(tree_b.root)

def sample_id_from_tree_b(tree_b, selected_b_ids):
	def dfs_b_sample_valid(node):
		if node.is_leaf():
			if node.selected:
				selected_b_ids.append(node.id)
		elif node.is_adj():
			if node.selected:
				selected_b_ids.append(node.id)
				return
			else:
				dfs_b_sample_valid(node.left)
				dfs_b_sample_valid(node.right)
		else:
			if node.selected:
				selected_b_ids.append(node.id)
	dfs_b_sample_valid(tree_b.root)
	
def sample_id_from_tree_a(tree_a, selected_a_ids, match_b_ids):
	def dfs_a_sample(node):
		if node.is_leaf():
			if random.randint(0,10) > 2:
				selected_a_ids.append(node.id)
				if node.match_id != 0:
					match_b_ids.append(node.match_id)
		elif node.is_adj():
			dfs_a_sample(node.left)
			dfs_a_sample(node.right)
		else:
			if random.randint(0,10) > 2:
				selected_a_ids.append(node.id)
				if node.match_id != 0:
					match_b_ids.append(node.match_id)
	dfs_a_sample(tree_a.root)

def get_correspondence_from_tree_a(tree_a, selected_a_ids, match_b_ids):
	def dfs_a_sample(node):
		if node.is_leaf():
			if node.match_id != 0:
				selected_a_ids.append(node.id)
				match_b_ids.append(node.match_id)
		elif node.is_adj():
			dfs_a_sample(node.left)
			dfs_a_sample(node.right)
		else:
			if node.match_id != 0:
				selected_a_ids.append(node.id)
				match_b_ids.append(node.match_id)
	dfs_a_sample(tree_a.root)

def clean_tree(node):
	node.loss = 999
	node.match_id = None
	node.selected = False
	if node.is_leaf():
		return
	elif node.is_adj():
		clean_tree(node.left)
		clean_tree(node.right)
	else:	
		return
		
if __name__ == '__main__':

	config = util.get_args()

	config.cuda = config.no_cuda
	if config.gpu < 0 and config.cuda:
		config.gpu = 0
	# torch.cuda.set_device(config.gpu)
	if config.cuda and torch.cuda.is_available():
		print("using CUDA on GPU ", config.gpu)
	else:
		print("Not using CUDA")
	encoder = torch.load('./models/vq_encoder_model_finetune.pkl', map_location=torch.device('cpu'))
	decoder = torch.load('./models/vq_decoder_model_finetune.pkl', map_location=torch.device('cpu'))
	model = GRASSMerge(config, encoder, decoder)
	model.cpu()
	model.eval()
	if config.finetune:
		print("fintune phase")

	grass_data = ChairDataset(dir=config.data_path, data_name='A')

	""" Specific configuration for Multi-Shape correspondences
	"""
	multi_shape_num = min(grass_data.data_size, 3)
	
	final_result = []
	for i in range(grass_data.data_size):

		# generating index -------------------------------------------------------------------------
		a_range = np.arange(grass_data.data_size)
		np.random.shuffle(a_range)
		# indices_array = [i, i+1, i+2, i+3, i+4]
		indices_array = [i]

		for j in range(a_range.shape[0]):
			if a_range[j] not in indices_array:
				indices_array.append(int(a_range[j]))
		indices_array = indices_array[:multi_shape_num]

		# assign the label for each tree -----------------------------------------------------------
		for g_idx in indices_array:
			tree = grass_data[g_idx]
			dfs_assign_label(tree.root)

		# use first frame as the anchor and iterate others 
		sel_indices = {}
		for j_idx in indices_array:
			sel_indices[j_idx] = []

		matches = []
		matches_node_indices = []

		anchor_idx = indices_array[0]
		for m_idx in indices_array[1:]:

			dfs_a(grass_data[m_idx].root, grass_data[ anchor_idx], model)

			selected_a_ids = []
			match_b_ids = []
			while len(selected_a_ids) < 1:
				sample_id_from_tree_a(grass_data[m_idx], selected_a_ids, match_b_ids)

			give_valid_to_tree_b(grass_data[anchor_idx], match_b_ids)
			selected_b_ids = []	  
			sample_id_from_tree_b(grass_data[anchor_idx], selected_b_ids)
			# show_correspondence(grass_data[anchor_idx], grass_data[m_idx])

			for b_id in selected_b_ids :
				if b_id not in sel_indices[anchor_idx]:
					sel_indices[anchor_idx].append(b_id)
					
			for a_id in selected_a_ids:
				if a_id not in sel_indices[m_idx]:
					sel_indices[m_idx].append(a_id)

			clean_tree(grass_data[anchor_idx].root)
			clean_tree(grass_data[m_idx].root)

			print('selected_a_ids', selected_a_ids)
			print('match_b_ids', match_b_ids)
			print('selected_b_ids', selected_b_ids)

		# # check if other shape has matches towards anchor idx
		# for m_idx in indices_array[1:]:

		# 	# dfs_a(grass_data[m_idx].root, grass_data[anchor_idx], model)
		# 	# # show_correspondence(grass_data[anchor_idx], grass_data[m_idx])

		# 	# selected_a_ids = []
		# 	# match_b_ids = []
		# 	# while len(selected_a_ids) < 1:
		# 	# 	get_correspondence_from_tree_a(grass_data[m_idx], selected_a_ids, match_b_ids)

		# 	# # add to match pairs
		# 	# for i in range(len(selected_a_ids)):
		# 	# 	matches.append([m_idx, anchor_idx])
		# 	# 	matches_node_indices.append([selected_a_ids[i], match_b_ids[i]])

		# 	dfs_a(grass_data[anchor_idx].root, grass_data[m_idx], model)

		# 	selected_a_ids = []
		# 	match_b_ids = []
		# 	while len(selected_a_ids) < 1:
		# 		get_correspondence_from_tree_a(grass_data[anchor_idx], selected_a_ids, match_b_ids)

		# 	# add to match pairs
		# 	for i in range(len(selected_a_ids)):
		# 		matches.append([anchor_idx, m_idx])
		# 		matches_node_indices.append([selected_a_ids[i], match_b_ids[i]])

		# check if other two has the same parts
		for j_idx in range(len(indices_array)):
			for m_idx in range(len(indices_array)):
				if j_idx == m_idx:
					continue
				
				j_idx_ = indices_array[j_idx]
				m_idx_ = indices_array[m_idx]

				dfs_a(grass_data[j_idx_].root, grass_data[m_idx_], model)
				
				selected_a_ids = []
				match_b_ids = []
				while len(selected_a_ids) < 1:
					get_correspondence_from_tree_a(grass_data[j_idx_], selected_a_ids, match_b_ids)
					
				clean_tree(grass_data[j_idx_].root)
				clean_tree(grass_data[m_idx_].root)

				# remove all matches if two matches is not equal
				if len(selected_a_ids) != len(match_b_ids):
					# for pair_idx in range(len(selected_a_ids)):
					# 	a_node_id = selected_a_ids[pair_idx]
					# 	if a_node_id in sel_indices[j_idx_]:
					# 		sel_indices[j_idx_].remove(a_node_id)
					# 		print('removed %d from set %d', a_node_id, j_idx_)
					
					# for pair_idx in range(len(match_b_ids)):
					# 	b_node_id = match_b_ids[pair_idx]
					# 	if b_node_id in sel_indices[m_idx_]:
					# 		sel_indices[m_idx_].remove(b_node_id)
					# 		print('removed %d from set %d', b_node_id, m_idx_)
					pass
				else:
					# check the pair is exist in sel_indices
					for pair_idx in range(min(len(selected_a_ids), len(match_b_ids))):
					# for pair_idx in range(len(selected_a_ids)):
						a_node_id = selected_a_ids[pair_idx]
						b_node_id = match_b_ids[pair_idx]
						if a_node_id in sel_indices[j_idx_] and b_node_id in sel_indices[m_idx_]:
							if len(sel_indices[j_idx_]) >= len(sel_indices[m_idx_]):
								sel_indices[j_idx_].remove(a_node_id)
								print('removed %d from set %d' % (a_node_id, j_idx_))
							elif len(sel_indices[j_idx_]) < len(sel_indices[m_idx_]):
								sel_indices[m_idx_].remove(b_node_id)
								print('removed %d from set %d' % (b_node_id, m_idx_))

		# remove matches
		for i in range(len(matches)):
			a_id = matches[i][0]
			b_id = matches[i][1]

			a_node_id = matches_node_indices[i][0]
			b_node_id = matches_node_indices[i][1]
			
			if a_node_id in sel_indices[a_id] and b_node_id in sel_indices[b_id]:
				sel_indices[a_id].remove(a_node_id)
				print('removed %d from set %d' % (a_node_id, a_id))

		# check if seat inside?
		seat_idx = 7
		seat_flag = False
		for idx in indices_array:
			if seat_idx in sel_indices[idx]:
				seat_flag = True
		
		if seat_flag == False:
			sel_indices[anchor_idx].append(seat_idx)

		shape_pair_ids = {}
		valid_shape_count = 0
		for j_idx in indices_array:
			if len(sel_indices[j_idx]) > 0:
				shape_pair_ids['shape_%d_index' % valid_shape_count] = j_idx
				shape_pair_ids['shape_%d_ids' % valid_shape_count] = sel_indices[j_idx]
				valid_shape_count += 1
		shape_pair_ids['valid_shapes'] = valid_shape_count
		print('shape_pair_ids', shape_pair_ids)
	
		final_result.append(shape_pair_ids)

		# tree_a = grass_data[i]
		# tree_b = grass_data[i+1]
		# #assign label
		# dfs_assign_label(tree_a.root)
		# dfs_assign_label(tree_b.root)
		
		# dfs_a(tree_a.root, tree_b, model)
		# #show_correspondence(tree_a, tree_b)
		# #sample_labels = random.sample(range(4), 2)
		# selected_a_ids = []
		# match_b_ids = []
		# while len(selected_a_ids) < 1:
		# 	sample_id_from_tree_a(tree_a, selected_a_ids, match_b_ids)
		# give_valid_to_tree_b(tree_b, match_b_ids)
		# selected_b_ids = []	  
		# sample_id_from_tree_b(tree_b, selected_b_ids)
		# print('selected_a_ids', selected_a_ids)
		# print('match_b_ids', match_b_ids)
		# print('selected_b_ids', selected_b_ids)
		# shape_pair_ids={'shape_a_index':i, 'shape_b_index':i+1, 'selected_a_ids': selected_a_ids, 'selected_b_ids':selected_b_ids}
		# final_result.append(shape_pair_ids)
	
	import pickle
	with open("shape_node_ids_%d_shapes.bin" % multi_shape_num, 'wb') as f:
		pickle.dump(final_result, f)
	savemat("shape_node_ids_%d_shapes.mat" % multi_shape_num, {'final_result':final_result})
	# boxes, labels = decode_structure(tree_a.root)
	# label_text = []
	# for label in labels:
		# if label == 0:
			# label_text.append('back')
		# elif label == 1:
			# label_text.append('seat')
		# elif label == 2:
			# label_text.append('leg')
		# elif label == 3:
			# label_text.append('armrest')

	# showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text)
	
	# boxes, labels = decode_structure(tree_b.root)
	# label_text = []
	# for label in labels:
		# if label == 0:
			# label_text.append('back')
		# elif label == 1:
			# label_text.append('seat')
		# elif label == 2:
			# label_text.append('leg')
		# elif label == 3:
			# label_text.append('armrest')

	# showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text)
	
		

	
	

	# example.addNoise()
	# boxes, labels = decode_structure(example.root, noise=True)
	# label_text = []
	# for label in labels:
		# if label == 0:
			# label_text.append('back')
		# elif label == 1:
			# label_text.append('seat')
		# elif label == 2:
			# label_text.append('leg')
		# elif label == 3:
			# label_text.append('armrest')
	# showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text)

	# refine_tree = example
	# for i in range(1):
		# refine_tree = inference(refine_tree)
		# boxes, labels = decode_structure(refine_tree.root)
		# label_text = []
		# for label in labels:
			# if label == 0:
				# label_text.append('back')
			# elif label == 1:
				# label_text.append('seat')
			# elif label == 2:
				# label_text.append('leg')
			# elif label == 3:
				# label_text.append('armrest')
		# showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text)

