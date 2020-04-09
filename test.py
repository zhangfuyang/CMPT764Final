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



def encode_fold(fold, tree):
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

    encoding = encode_node(tree.root)
    return encoding

def decode_fold(model, feature, tree, Boxes, Syms, Labels):
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

    def decode_node(feature, node, newBoxs, newSyms, newLabels):
        if node.is_leaf():
            f = encode_node(node)
            reBox = model.boxNode(feature, f)
            #new_node = Tree.Node(box=reBox, node_type=Tree.NodeType.BOX)
            newBoxs.append(reBox.detach().cpu())
            newLabels.append(node.box_label)

        elif node.is_adj():
            d = model.deSampleLayer(feature)
            feature = model.vqlizationFeature(feature)
            left_node = node.left
            right_node = node.right
            fl = encode_node(left_node)
            fr = encode_node(right_node)
            left_f = model.outterNode(feature, fr)
            right_f = model.outterNode(feature, fl)
            decode_node(left_f, node.left, newBoxs, newSyms, newLabels)
            decode_node(right_f, node.right, newBoxs, newSyms, newLabels)
        elif node.is_sym():
            feature = model.vqlizationFeature(feature)

            f = encode_node(node)
            new_f, sym_f = model.symParaNode(feature, f)
            newSyms.append(sym_f.detach().cpu())
            decode_node(new_f, node.left, newBoxs, newSyms, newLabels)

    decode_node(feature, tree.root, Boxes, Syms, Labels)

grass_data = ChairDataset(config.data_path)
def my_collate(batch):
    return batch
test_iter = torch.utils.data.DataLoader(grass_data, batch_size=1,
                                        shuffle=False, collate_fn=my_collate)

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


for batch_idx, batch in enumerate(test_iter):
    print(batch_idx)
    example=batch[0]
    boxes, labels = decode_structure(example.root)
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

    showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text)

    refine_tree = example
    for i in range(1):
        refine_tree = inference(refine_tree)
        boxes, labels = decode_structure(refine_tree.root)
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

        showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text)

    example.addNoise()
    boxes, labels = decode_structure(example.root, noise=True)
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
    showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text)

    refine_tree = example
    for i in range(1):
        refine_tree = inference(refine_tree)
        boxes, labels = decode_structure(refine_tree.root)
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
        showGenshape(torch.cat(boxes,0).data.cpu().numpy(), labels=label_text)

