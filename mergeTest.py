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
from render2mesh import directRender, alignBoxAndRender
from scipy.io import loadmat
import os
import numpy as np


config = util.get_args()
config.cuda = not config.no_cuda
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
if config.finetune:
    print("fintune phase")

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
    objnames = []
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
        if node.is_leaf():  # BOX
            if noise:
                reBox = node.box_noise
            else:
                reBox = node.box
            reBoxes = [reBox]
            reLabels = [node.box_label]
            reObj = [node.objname]
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
                    reObj.append(node.objname)
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
                    reObj.append(node.objname)
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
                reObj.append(node.objname)

            boxes.extend(reBoxes)
            labels.extend(reLabels)
            objnames.extend(reObj)
    return boxes, labels, objnames


def encode_fold(fold, root):
    def encode_node(node):
        if node.is_leaf():
            if config.finetune:
                n = fold.add('leafNode', node.box_noise)
            else:
                n = fold.add('leafNode', node.box)
            return n
        if node.is_adj():
            left = encode_node(node.left)
            right = encode_node(node.right)
            n = fold.add('adjNode', left, right)
            return n
        if node.is_sym():
            feature = encode_node(node.left)
            if config.finetune:
                n = fold.add('symNode', feature, node.sym_noise)
            else:
                n = fold.add('symNode', feature, node.sym)
            return n

    encoding = encode_node(root)
    return encoding

def decode_fold(model, feature, root, Boxes, Syms, Labels, Ops, objnames):
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

    def decode_node(feature, node, newBoxs, newSyms, newLabels, newOps, newObj):
        if node.is_leaf():
            f = encode_node(node)
            reBox = model.boxNode(feature, f)
            #new_node = Tree.Node(box=reBox, node_type=Tree.NodeType.BOX)
            newBoxs.append(reBox.detach().cpu())
            newLabels.append(node.box_label)
            newOps.append(0)
            newObj.append(node.objname)
            return 0

        elif node.is_adj():
            d = model.deSampleLayer(feature)
            node_loss, feature = model.vqlizationWithLoss(feature)
            left_node = node.left
            right_node = node.right
            fl = encode_node(left_node)
            fr = encode_node(right_node)
            left_f = model.outterNode(feature, fr)
            right_f = model.outterNode(feature, fl)
            left_loss = decode_node(left_f, node.left, newBoxs, newSyms, newLabels, newOps, newObj)
            right_loss = decode_node(right_f, node.right, newBoxs, newSyms, newLabels, newOps, newObj)
            newOps.append(1)
            return left_loss + right_loss + node_loss
        elif node.is_sym():
            node_loss, feature = model.vqlizationWithLoss(feature)
            f = encode_node(node)
            new_f, sym_f = model.symParaNode(feature, f)
            newSyms.append(sym_f.detach().cpu())
            left_loss = decode_node(new_f, node.left, newBoxs, newSyms, newLabels, newOps, newObj)
            newOps.append(2)
            return left_loss + node_loss

    loss = decode_node(feature, root, Boxes, Syms, Labels, Ops, objnames)
    return loss

def my_collate(batch):
    return batch

def inference(root):
    enc_fold = FoldExt(cuda=config.cuda)
    enc_fold_nodes = []
    enc_fold_nodes.append(encode_fold(enc_fold, root))
    enc_fold_nodes = enc_fold.apply(model, [enc_fold_nodes])
    enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)
    refineboxes = []
    syms = []
    Labels = []
    Ops = []
    Objs = []
    loss = decode_fold(model, enc_fold_nodes[0], root, refineboxes, syms, Labels, Ops, Objs)
    refineboxes = torch.cat(refineboxes, 0)
    refineLabels = torch.Tensor(Labels)

    refineOps = torch.Tensor(Ops).unsqueeze(0)
    if len(syms) == 0:
        syms = torch.zeros((1, 8))
    else:
        syms = torch.cat(syms, 0)
    refine_tree = Tree(refineboxes, refineOps, syms, refineLabels, Objs)
    #return refine_tree
    return loss, refine_tree


def unique(list): 
    x = np.array(list) 
    return np.unique(x).tolist() 

def get_labels(node):
    if node.is_leaf():
        return [node.box_label]
    elif node.is_adj():
        return unique(get_labels(node.left) + get_labels(node.right))
    else:
        return get_labels(node.left) 

def get_all_labels(tree, nodeIds):
    def dfs(node, nodeIds, all_labels, nodes):
        if node.id in nodeIds:
            l = get_labels(node)
            all_labels = all_labels.append(l)
            nodes = nodes.append(node)
            return
        if node.is_adj():
            if node.left is not None:
                dfs(node.left, nodeIds, all_labels, nodes)
            if node.right is not None:
                dfs(node.right, nodeIds, all_labels, nodes)
        elif node.is_sym():
            if node.left is not None:
                dfs(node.left, nodeIds, all_labels, nodes)

    all_labels = []
    nodes = []
    dfs(tree.root, nodeIds, all_labels, nodes)
    return nodes, all_labels

def EqualSymPara(s1, s2):
    s1 = s1[0]
    s2 = s2[0]
    if s1[0] != s2[0]:
        return False
    # dir_1 = s1[1:4]
    # dir_2 = s2[1:4]
    # print (abs(torch.dot(dir_1, dir_2)))
    if s1[0] == -1:
        dir_1 = s1[1:4]
        dir_2 = s2[1:4]
        if abs(torch.dot(dir_1, dir_2)) >0.8:
            return True
    
    if s1[0] == 1:
        dir_1 = s1[1:4]
        dir_2 = s2[1:4]
        p1 = s1[4:7]
        p2 = s2[4:7]
        if abs(torch.dot(dir_1, dir_2)) >0.8 and torch.dot(dir_1, p2-p1) < 0.1:
            return True
    return False

def merge_syms(label, all_nodes, all_labels):
    merged = 1
    while merged > 0:
        merged = 0
        i = 0
        while i < len(all_nodes):
            if all_nodes[i].is_sym() and label in all_labels[i] :
                j = i+1
                while j < len(all_nodes):
                    if all_nodes[j].is_sym() and label in all_labels[j]:
                        if EqualSymPara(all_nodes[i].sym, all_nodes[j].sym):
                            right_node = all_nodes.pop(j)
                            right_labels = all_labels.pop(j)
                            left_node = all_nodes.pop(i)
                            left_lables = all_labels.pop(i)
                            adj = Tree.Node(left=left_node.left, right=right_node.left, node_type=Tree.NodeType.ADJ)
                            all_nodes.append(Tree.Node(left=adj, sym=right_node.sym, node_type=Tree.NodeType.SYM))
                            all_labels.append(unique(right_labels + left_lables))
                            merged = merged + 1
                            j = i
                    j = j+1
            i = i+1
    return all_nodes, all_labels

def merge_labels (label1, label2, all_nodes, all_labels):
    merged = 1
    while merged > 0:
        merged = 0
        i = 0
        while i < len(all_nodes):
            if label1 in all_labels[i] or label2 in all_labels[i]:
                j = i+1
                while j < len(all_nodes):
                    if label1 in all_labels[j] or label2 in all_labels[j]:
                        right_node = all_nodes.pop(j)
                        right_labels = all_labels.pop(j)
                        left_node = all_nodes.pop(i)
                        left_lables = all_labels.pop(i)
                        all_nodes.append(Tree.Node(left=left_node, right=right_node, node_type=Tree.NodeType.ADJ))
                        all_labels.append(unique(right_labels + left_lables))
                        merged = merged + 1
                        j = i
                    j = j+1
            i = i+1
    return all_nodes, all_labels


def get_root_merge_trees(trees, nodeIds):
    all_nodes = []
    all_labels = []
    for i in range(len(trees)):
        nodes, labels = get_all_labels(trees[i], nodeIds[i])
        all_nodes = all_nodes + nodes
        all_labels = all_labels + labels
    
    for i in range(4):
        all_nodes, all_labels = merge_syms(i, all_nodes, all_labels)
    for i in range(4):
        all_nodes, all_labels = merge_labels(i, i, all_nodes, all_labels)
    all_nodes, all_labels = merge_labels(1, 3, all_nodes, all_labels)
    all_nodes, all_labels = merge_labels(1, 2, all_nodes, all_labels)
    all_nodes, all_labels = merge_labels(1, 0, all_nodes, all_labels)

    assert len(all_nodes) == 1
    return all_nodes[0]

def get_label_text(labels):
    label_text = []
    for label in labels:
        if label == 0:
            label_text.append('back')
        elif label == 1:
            label_text.append('seat')
        elif label == 2:
            label_text.append('leg')
        elif label == 3:
            label_text.append('armrest')
    return label_text

    
grass_data = ChairDataset(config.data_path)
file_name = 'shape_node_ids_5_shapes.mat'
pairs = loadmat(os.path.join('./', file_name))['final_result']
image = True
pairs = pairs[0]
for i in range(len(pairs)):
    print (i)
    num = pairs[i]['valid_shapes'][0][0][0][0]
    node_arrays = []
    tree_array = []
    for j in range(num):
        idx = pairs[i]['shape_' + str(j) +'_index'][0][0][0][0]
        nodes = pairs[i]['shape_' + str(j) +'_ids'][0][0][0]
        
        tree = grass_data[idx]
        node_arrays.append(nodes)
        tree_array.append(tree)
        boxes, labels, objnames = decode_structure(tree.root)
        label_text = get_label_text(labels)
        showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text,
                        save=image, savedir='demo/Sample_' + str(i)+'_Original_Shape_' + str(j) + '.png')
       

    merge_root = get_root_merge_trees(tree_array, node_arrays)
    boxes, labels, objnames = decode_structure(merge_root)
    label_text = get_label_text(labels)
    showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text,
                    save=image, savedir='demo/Sample_' + str(i)+'_Original_Merge.png')
    
    refine_root = merge_root

    j = 0
    prev_loss = 0.0
    loss = 1000.0
    treshold = 0.005
    while j==0 or prev_loss - loss > treshold:
        j = j+1
        prev_loss = loss
        loss, refine_tree = inference(refine_root)
        refine_root = refine_tree.root
        boxesRefine, labelsRefine, objnamesRefine = decode_structure(refine_root)
        label_text = get_label_text(labelsRefine)
        showGenshape(torch.cat(boxesRefine,0).data.cpu().numpy(), labels=label_text,
                        save=image, savedir='demo/Sample_' + str(i) + '_Refine_Merge_try_' + str(j) + '.png')
