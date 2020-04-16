import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import os
import numpy as np
from draw3dOBB import showGenshape

def give_node_id(root):
    def dfs_give_id(node):
        global id
        node.id = id
        id = id + 1
        #find which node of tree A need to be dealt with
        if node.is_leaf():
            return
        elif node.is_adj():
            dfs_give_id(node.left)
            dfs_give_id(node.right)
        else:
            dfs_give_id(node.left)
    
    global id 
    id = 0
    dfs_give_id(root)
    
class Tree(object):
    class NodeType(Enum):
        BOX = 0     # box node
        ADJ = 1     # adjacency (adjacent part assembly) node
        SYM = 2     # symmetry (symmetric part grouping) node

    class Node(object):
        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None, box_label=None, objname=None, tree_id=None):
            self.box = box            # box feature vector for a leaf node
            self.box_noise = box          # box feature vector for a leaf node with noise
            self.sym = sym            # symmetry parameter vector for a symmetry node
            self.sym_noise = sym          # symmetry parameter vector for a symmetry node with noise
            self.left = left        # left child for ADJ or SYM (a symmeter generator)
            self.right = right        # right child
            self.node_type = node_type
            self.objname = objname
            self.label = torch.LongTensor([self.node_type.value])
            self.box_label = box_label
            self.id = id
            self.loss = 999
            self.match_id = None
            self.selected = False
            self.tree_id = tree_id

        def is_leaf(self):
            return self.node_type == Tree.NodeType.BOX and self.box is not None

        def is_adj(self):
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            return self.node_type == Tree.NodeType.SYM

    def __init__(self, boxes, ops, syms, labels, objname):
        box_list = [b for b in torch.split(boxes, 1, 0)]
        sym_param = [s for s in torch.split(syms, 1, 0)]
        labels_list = list(torch.split(labels, 1))
        box_list.reverse()
        sym_param.reverse()
        labels_list.reverse()
        objname.reverse()
        queue = []
        self.leves = []
        self.symNodes = []
        self.adjNode = []
        self.ops = ops
        for id in range(ops.size()[1]):
            if ops[0, id] == Tree.NodeType.BOX.value:
                n = Tree.Node(box=box_list.pop(), node_type=Tree.NodeType.BOX, box_label=labels_list.pop().numpy()[0], objname=objname.pop())
                self.leves.append(n)
                queue.append(n)
            elif ops[0, id] == Tree.NodeType.ADJ.value:
                left_node = queue.pop()
                right_node = queue.pop()
                n = Tree.Node(left=left_node, right=right_node, node_type=Tree.NodeType.ADJ)
                self.adjNode.append(n)
                queue.append(n)
            elif ops[0, id] == Tree.NodeType.SYM.value:
                node = queue.pop()
                n = Tree.Node(left=node, sym=sym_param.pop(), node_type=Tree.NodeType.SYM)
                self.symNodes.append(n)
                queue.append(n)
        assert len(queue) == 1
        self.root = queue[0]

    def addNoise(self):
        z = min(5, len(self.leves)+len(self.symNodes))
        number = np.random.randint(1, z)
        index = np.random.choice(range(len(self.leves)+len(self.symNodes)), number)
        base = np.zeros(len(self.leves) + len(self.symNodes))
        base[index] = 1
        base_leves = base[:len(self.leves)]
        base_sym = base[len(self.leves):]
        for i in range(len(self.leves)):
            box = self.leves[i].box
            s = box.size()[0]
            noise1 = box.new(s, 3).normal_(0, 0.08) * base_leves[i]
            noise2 = box.new(s, 3).normal_(0, 0.03) * base_leves[i]
            noise3 = box.new(s, 3).normal_(0, 0.08) * base_leves[i]
            noise4 = box.new(s, 3).normal_(0, 0.03) * base_leves[i]
            noise = torch.cat((noise1, noise2, noise3, noise4), 1)
            self.leves[i].box_noise = box+noise

        for i in range(len(self.symNodes)):
            sym = self.symNodes[i].sym
            noise = sym.new(sym.size()).normal_(0, 0.05) * base_sym[i]
            self.symNodes[i].sym_noise = sym + noise

class ChairDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        box_data = torch.from_numpy(loadmat(os.path.join(self.dir, 'C_chair_boxes.mat'))['boxes']).float()
        op_data = torch.from_numpy(loadmat(os.path.join(self.dir, 'C_chair_ops.mat'))['ops']).float()
        sym_data = torch.from_numpy(loadmat(os.path.join(self.dir, 'C_chair_syms.mat'))['syms']).float()
        label_data = torch.from_numpy(loadmat(os.path.join(self.dir, 'C_chair_labels.mat'))['labels']).float()
        objname = loadmat(os.path.join(self.dir, 'C_chair_objname.mat'))['obj'][0]
        num_examples = op_data.size()[1]
        box_data = torch.chunk(box_data, num_examples, 1)
        op_data = torch.chunk(op_data, num_examples, 1)
        sym_data = torch.chunk(sym_data, num_examples, 1)

        self.transform = transform
        self.trees = []
        for i in range(len(op_data)):
            boxes = torch.t(box_data[i])
            ops = torch.t(op_data[i])
            box_num = int((ops==0).sum())
            syms = torch.t(sym_data[i])
            labels = label_data[:, i]
            tree = Tree(boxes, ops, syms, labels, [objname[i][0][box_i][0] for box_i in range(box_num)])
            give_node_id(tree.root)
            self.trees.append(tree)
            #showGenshape(boxes.data.cpu().numpy())

        return

    def __getitem__(self, index):
        return self.trees[index]

    def __len__(self):
        return len(self.trees)