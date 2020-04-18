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
from scipy.io import savemat
from itertools import combinations
import os

def vrrotvec2mat(rotvector):
	s = math.sin(rotvector[3])
	c = math.cos(rotvector[3])
	t = 1 - c
	x = rotvector[0]
	y = rotvector[1]
	z = rotvector[2]
	m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]])
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

def find_node_num(node):
	if node.is_leaf():
		#assign loss to each node of tree b
		return 0
	elif node.is_adj():
		left_num = find_node_num(node.left)
		right_num = find_node_num(node.right)
		return left_num + right_num + 1
	else:
		left_num = find_node_num(node.left)
		return left_num + 1
		
def find_correspondence_loss(node_a, tree_b, model):
	def dfs_b(node):
		if node.is_adj():
			#check left child
			#print('node_a label, ', node_a.box_label)
			#print('node.left label, ', node.left.box_label)
			if node.left.box_label == node_a.box_label:
				print('replace node id of b: ', node.left.id)
				print('replace node type of b: ', node.left.node_type)
				temp = node.left
				node.left = node_a
				#get error
				loss_node_num = find_node_num(tree_b.root)
				root_feature = encode_tree(model, tree_b)
				loss = decode_tree(model, root_feature, tree_b)
				#change to original
				node.left = temp
				node.left.loss = loss
				print('replace loss : ', loss)
				print('replace loss ava : ', node.left.loss)
			if node.left.is_adj():
				dfs_b(node.left)
			#print('node_a label, ', node_a.box_label)
			#print('node.right label, ', node.right.box_label)
			if node.right.box_label == node_a.box_label:
				print('replace node id of b: ', node.right.id)
				print('replace node type of b: ', node.right.node_type)
				temp = node.right
				node.right = node_a
				#get error
				loss_node_num = find_node_num(tree_b.root)
				root_feature = encode_tree(model, tree_b)
				loss = decode_tree(model, root_feature, tree_b)
				node.right = temp
				node.right.loss = loss
				print('replace loss : ', loss)
				print('replace loss ava : ', node.right.loss)
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
			label_text.append('shape_a_part')
			for i in range(box_b.size(0)):
				label_text.append('shape_b_part')
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
				label_text.append('shape_a_part')
			for i in range(box_b.size(0)):
				label_text.append('shape_b_part')
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
			if random.randint(0,10) > 5:
				selected_a_ids.append(node.id)
				if node.match_id != 0:
					match_b_ids.append(node.match_id)
		elif node.is_adj():
			dfs_a_sample(node.left)
			dfs_a_sample(node.right)
		else:
			if random.randint(0,10) > 5:
				selected_a_ids.append(node.id)
				if node.match_id != 0:
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

def find_all_node_num(node):
	if node.is_leaf():
		#assign loss to each node of tree b
		return 1
	elif node.is_adj():
		left_num = find_node_num(node.left)
		right_num = find_node_num(node.right)
		return left_num + right_num + 1
	else:
		left_num = find_node_num(node.left)
		return left_num + 1
		
if __name__ == '__main__':

	config = util.get_args()
	config.cuda = config.no_cuda
	if config.gpu < 0 and config.cuda:
		config.gpu = 0
	torch.cuda.set_device(config.gpu)
	if config.cuda and torch.cuda.is_available():
		print("using CUDA on GPU ", config.gpu)
	else:
		print("Not using CUDA")
	encoder = torch.load('./models/vq_encoder_model_finetune.pkl')
	decoder = torch.load('./models/vq_decoder_model_finetune.pkl')
	model = GRASSMerge(config, encoder, decoder)
	model.cpu()
	model.eval()
	if config.finetune:
		print("fintune phase")
	
	result_path = './result/'+config.testset
	if not os.path.exists(result_path):
		os.makedirs(result_path)
		
	grass_data = ChairDataset(config.data_path, data_name=config.testset)
	
	iters = combinations(list(range(grass_data.data_size)), 2)
	
	final_result = []
	count = 0
	for it in iters:
		trees = []
		for idx in it:
			trees.append(grass_data[idx])
		#assign label
		num_0 = find_all_node_num(trees[0].root)
		num_1 = find_all_node_num(trees[1].root)
		if num_0 > num_1:
			tree_a = trees[1]
			tree_b = trees[0]
		else:
			tree_a = trees[0]
			tree_b = trees[1]
		dfs_assign_label(tree_a.root)
		dfs_assign_label(tree_b.root)
		
		dfs_a(tree_a.root, tree_b, model)
		# if count == 0:
			# boxes_a, labels_a = decode_structure(tree_a.root)
			# label_text = []
			# for label in labels_a:
				# if label == 0:
					# label_text.append('back')
				# elif label == 1:
					# label_text.append('seat')
				# elif label == 2:
					# label_text.append('leg')
				# elif label == 3:
					# label_text.append('armrest')
			# showGenshape(torch.cat(boxes_a,0).data.cpu().numpy(), labels = label_text)
			# boxes_b, labels_b = decode_structure(tree_b.root)
			# label_text = []
			# for label in labels_b:
				# if label == 0:
					# label_text.append('back')
				# elif label == 1:
					# label_text.append('seat')
				# elif label == 2:
					# label_text.append('leg')
				# elif label == 3:
					# label_text.append('armrest')
			
			# showGenshape(torch.cat(boxes_b,0).data.cpu().numpy(), labels = label_text)
			# show_correspondence(tree_a, tree_b)
		   
		#sample_labels = random.sample(range(4), 2)
		selected_a_ids = []
		match_b_ids = []
		while len(selected_a_ids) < 1:
			sample_id_from_tree_a(tree_a, selected_a_ids, match_b_ids)
		give_valid_to_tree_b(tree_b, match_b_ids)
		selected_b_ids = []	  
		sample_id_from_tree_b(tree_b, selected_b_ids)
		print('selected_a_ids', selected_a_ids)
		print('match_b_ids', match_b_ids)
		print('selected_b_ids', selected_b_ids)
		
		if(len(selected_b_ids) == 0):
			continue
		shape_pair_ids = {}
		shape_pair_ids['shape_%d_index' % 0] = tree_a.id
		shape_pair_ids['shape_%d_ids' % 0] = selected_a_ids
		shape_pair_ids['shape_%d_index' % 1] = tree_b.id
		shape_pair_ids['shape_%d_ids' % 1] = selected_b_ids
		shape_pair_ids['valid_shapes'] = 2
		print('shape_pair_ids', shape_pair_ids)
	
		final_result.append(shape_pair_ids)
		
		# shape_pair_ids={'shape_a_index':i, 'shape_b_index':i+1, 'selected_a_ids': selected_a_ids, 'selected_b_ids':selected_b_ids}
		# final_result.append(shape_pair_ids)
		clean_tree(tree_a.root)
		clean_tree(tree_b.root)
		count += 1
		
	savemat(result_path + "/shape_node_ids_2_shapes.mat", {'final_result':final_result})
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

