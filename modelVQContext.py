import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
from dataset import Tree
from random import randint
from random import shuffle
from ObjEditer import ShapeGeometry

#########################################################################################
# Encoder
#########################################################################################


class BoxEncoder(nn.Module):

    def __init__(self, boxSize, featureSize, hiddenSize):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(boxSize, featureSize)
        self.middlein = nn.Linear(featureSize, hiddenSize)
        self.middleout = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, boxes_in):
        boxes = self.encoder(boxes_in)
        boxes = self.tanh(boxes)
        boxes = self.middlein(boxes)
        boxes = self.tanh(boxes)
        boxes = self.middleout(boxes)
        boxes = self.tanh(boxes)
        return boxes


class AdjEncoder(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(AdjEncoder, self).__init__()
        self.left = nn.Linear(featureSize, hiddenSize)
        self.right = nn.Linear(featureSize, hiddenSize, bias=False)
        self.second = nn.Linear(hiddenSize, hiddenSize)
        self.third = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, left_in, right_in):
        out = self.left(left_in)
        out += self.right(right_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        out = self.third(out)
        out = self.tanh(out)
        return out


class SymEncoder(nn.Module):

    def __init__(self, featureSize, symmetrySize, hiddenSize):
        super(SymEncoder, self).__init__()
        self.left = nn.Linear(featureSize, hiddenSize)
        self.right = nn.Linear(symmetrySize, hiddenSize)
        self.second = nn.Linear(hiddenSize, hiddenSize)
        self.third = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, left_in, right_in):
        out = self.left(left_in)
        out += self.right(right_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        out = self.third(out)
        out = self.tanh(out)
        return out


class GRASSEncoder(nn.Module):

    def __init__(self, config):
        super(GRASSEncoder, self).__init__()
        self.boxEncoder = BoxEncoder(
            boxSize=config.boxSize, featureSize=config.featureSize, hiddenSize=config.hiddenSize)
        self.adjEncoder = AdjEncoder(
            featureSize=config.featureSize, hiddenSize=config.hiddenSize)
        self.symEncoder = SymEncoder(
            featureSize=config.featureSize, symmetrySize=config.symmetrySize, hiddenSize=config.hiddenSize)

        self.dict = nn.Embedding(
            num_embeddings=config.vqDictionary, embedding_dim=config.vqFeature)

        self.num_embeddings = config.vqDictionary
        self.embedding_dim = config.vqFeature
        self.featureLength = config.featureSize

#########################################################################################
# Decoder
#########################################################################################


class NodeClassifier(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(NodeClassifier, self).__init__()
        self.first = nn.Linear(featureSize, hiddenSize)
        self.tanh = nn.Tanh()
        self.second = nn.Linear(hiddenSize, 3)
        self.softmax = nn.Softmax()

    def forward(self, feature):
        out = self.first(feature)
        out = self.tanh(out)
        out = self.second(out)
        out = self.softmax(out)
        return out


class Desampler(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(Desampler, self).__init__()
        self.mlp1 = nn.Linear(featureSize, hiddenSize)
        self.mlp2 = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.tanh(self.mlp1(input))
        output = self.tanh(self.mlp2(output))
        return output


class AdjDecoder(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(AdjDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, hiddenSize)
        self.second = nn.Linear(hiddenSize, hiddenSize)
        self.left = nn.Linear(hiddenSize, featureSize)
        self.right = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, parent_in):
        out = self.decode(parent_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        l = self.left(out)
        r = self.right(out)
        l = self.tanh(l)
        r = self.tanh(r)
        return l, r


class SymDecoder(nn.Module):

    def __init__(self, featureSize, symmetrySize, hiddenSize):
        super(SymDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, hiddenSize)
        self.second = nn.Linear(hiddenSize, hiddenSize)
        self.tanh = nn.Tanh()
        self.left = nn.Linear(hiddenSize, featureSize)
        self.right = nn.Linear(hiddenSize, symmetrySize)

    def forward(self, parent_in):
        out = self.decode(parent_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        f = self.left(out)
        f = self.tanh(f)
        s = self.right(out)
        s = self.tanh(s)
        return f, s


class BoxDecoder(nn.Module):

    def __init__(self, boxSize, featureSize, hiddenSize):
        super(BoxDecoder, self).__init__()
        self.first = nn.Linear(featureSize, hiddenSize)
        self.decode = nn.Linear(hiddenSize, boxSize)
        self.tanh = nn.Tanh()

    def forward(self, parent_in):
        out = self.first(parent_in)
        out = self.tanh(out)
        out = self.decode(out)
        out = self.tanh(out)
        return out


class MergeTwoCode(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(MergeTwoCode, self).__init__()
        self.left = nn.Linear(featureSize, hiddenSize)
        self.right = nn.Linear(featureSize, hiddenSize, bias=False)
        self.second = nn.Linear(hiddenSize, hiddenSize)
        self.third = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, left_in, right_in):
        out = self.left(left_in)
        out += self.right(right_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        out = self.third(out)
        out = self.tanh(out)
        return out


class GRASSDecoder(nn.Module):
    def __init__(self, config):
        super(GRASSDecoder, self).__init__()
        self.boxDecoder = BoxDecoder(
            boxSize=config.boxSize, featureSize=config.featureSize, hiddenSize=config.hiddenSize)
        self.symDecoder = SymDecoder(
            featureSize=config.featureSize, symmetrySize=config.symmetrySize, hiddenSize=config.hiddenSize)

        self.adjDecoderIncomplete = AdjDecoder(
            featureSize=config.featureSize*2, hiddenSize=config.hiddenSize*2)
        self.boxDecoderIncomplete = BoxDecoder(
            boxSize=config.boxSize, featureSize=config.featureSize*2, hiddenSize=config.hiddenSize*2)
        self.symDecoderIncomplete = SymDecoder(
            featureSize=config.featureSize*2, symmetrySize=config.symmetrySize, hiddenSize=config.hiddenSize*2)
        self.nodeClassifierIncomplete = NodeClassifier(
            featureSize=config.featureSize*2, hiddenSize=config.hiddenSize)

        self.adjDecoderRedundant = AdjDecoder(
            featureSize=config.featureSize*2, hiddenSize=config.hiddenSize*2)
        self.boxDecoderRedundant = BoxDecoder(
            boxSize=config.boxSize, featureSize=config.featureSize*2, hiddenSize=config.hiddenSize*2)
        self.symDecoderRedundant = SymDecoder(
            featureSize=config.featureSize*2, symmetrySize=config.symmetrySize, hiddenSize=config.hiddenSize*2)
        self.nodeClassifierRedundant = NodeClassifier(
            featureSize=config.featureSize*2, hiddenSize=config.hiddenSize)

        self.desampler = Desampler(
            featureSize=config.featureSize, hiddenSize=config.hiddenSize)
        self.getOutter = MergeTwoCode(
            featureSize=config.featureSize, hiddenSize=config.hiddenSize)
        self.mergeOutter = MergeTwoCode(
            featureSize=config.featureSize, hiddenSize=config.hiddenSize)

    def classLossLayerIncomplete(self, f1, f2):
        f = self.nodeClassifierIncomplete(f1)
        return torch.log(f).mul(f2).sum(1).mul(-0.2)

    def classLayerIncomplete(self, f):
        l = self.nodeClassifierIncomplete(f)
        _, op = torch.max(l, 1)
        return op

    def classLossLayerRedundant(self, f1, f2):
        f = self.nodeClassifierRedundant(f1)
        return torch.log(f).mul(f2).sum(1).mul(-0.2)

    def classLayerRedundant(self, f):
        l = self.nodeClassifierRedundant(f)
        _, op = torch.max(l, 1)
        return op

#########################################################################################
# Utility
#########################################################################################


def vrrotvec2mat_cpu(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x *
                                                             y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]])
    return m


def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z,
                                                             t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]).cuda()
    return m

#########################################################################################
# Merge training operation
#########################################################################################

class GRASSMerge(nn.Module):
    def __init__(self, config, encoder, decoder):
        super(GRASSMerge, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    ########################################################
    # Encoder Related
    ########################################################

    def leafNodeTest(self, box):
        return self.encoder.boxEncoder(box)

    def adjNodeTest(self, left, right):
        return self.encoder.adjEncoder(left, right)

    def symNodeTest(self, feature, sym):
        return self.encoder.symEncoder(feature, sym)

    def symNodeFeature(self, feature, sym):
        return self.encoder.symEncoder(feature, sym)

    def leafNode(self, box):
        return self.encoder.boxEncoder(box)

    def leafNodeWithNoise(self, box):
        s = box.size()[0]
        noise1 = Variable(box.data.new(s, 3).normal_(0, 0.08))
        noise2 = Variable(box.data.new(s, 3).normal_(0, 0.03))
        noise3 = Variable(box.data.new(s, 3).normal_(0, 0.08))
        noise4 = Variable(box.data.new(s, 3).normal_(0, 0.03))
        noise = torch.cat((noise1, noise2, noise3, noise4), 1)
        return self.encoder.boxEncoder(box + noise)

    def adjNode(self, left, right):
        return self.encoder.adjEncoder(left, right)

    def adjNodeWithNoise(self, left, right):
        noiseL = Variable(left.data.new(left.size()).normal_(0, 0.05))
        noiseR = Variable(right.data.new(right.size()).normal_(0, 0.05))
        return self.encoder.adjEncoder(left + noiseL, right + noiseR)

    def symNode(self, feature, sym):
        return self.encoder.symEncoder(feature, sym)

    def symNodeWithNoise(self, feature, sym):
        #noiseL = Variable(feature.data.new(feature.size()).normal_(0, 0.05))
        noiseS = Variable(sym.data.new(sym.size()).normal_(0, 0.05))
        return self.encoder.symEncoder(feature, sym + noiseS)

    ########################################################
    # Decoder Related
    ########################################################

    def deSampleLayer(self, feature):
        return self.decoder.desampler(feature)

    def outterNode(self, outter, inner):
        return self.decoder.getOutter(outter, inner)

    def boxNode(self, outter, inner):
        feature = self.decoder.mergeOutter(outter, inner)
        return self.decoder.boxDecoder(feature)

    def symParaNode(self, outter, inner):
        feature = self.decoder.mergeOutter(outter, inner)
        return self.decoder.symDecoder(feature)

    ########################################################
    # Utility Related
    ########################################################

    def mseBoxLossLayer(self, f1, f2):
        loss = ((f1.add(-f2))**2).sum(1).mul(0.4)
        return loss

    def mseSymLossLayer(self, f1, f2):
        loss = ((f1.add(-f2))**2).sum(1).mul(3)
        return loss

    def addLayer(self, f1, f2):
        return f1.add_(f2)

    def contrastiveLayer(self, f1, f2, f3):
        fa = F.relu(f1.mul(2).add_(-f2))
        fb = F.relu(f3.mul(2).add_(-f2))
        return fa + fb

    def concat(self, feature, inner):
        return torch.cat((feature, inner), 1)

    def lossPostProcessing(self, loss1, loss2):
        l = torch.cat((loss1.unsqueeze(1), loss2.unsqueeze(1)), 1)
        l = torch.min(l, 1)[1].mul(9).add(1).type(torch.cuda.FloatTensor)
        return loss2.mul(l)

    def cat4LossGeneral(self, vqLoss, loss, claLoss, eLoss):
        vqLoss = vqLoss.unsqueeze(1)
        loss = loss.unsqueeze(1)
        claLoss = claLoss.unsqueeze(1)
        eLoss = eLoss.unsqueeze(1)
        return torch.cat((loss, vqLoss, claLoss, eLoss), 1)

    def node_add_f(self, f, node):
        node.encode_f = f

    def vectorAdder(self, v1, v2):
        return v1.add_(v2)

    def tensor_cat(self, x, y):
        return torch.cat((x,y))

    def zero_tensor(self, f):
        t = torch.zeros(f.size()[0], dtype=torch.float)
        if self.config.cuda:
            t = t.cuda()
        return t
    ########################################################
    # VQ Related
    ########################################################
    def vqlizationLoss(self, feature):
        f = feature.view(-1, self.encoder.embedding_dim)
        W = self.encoder.dict.weight

        def L2_dist(a, b):
            return ((a - b) ** 2)

        j = L2_dist(f[:, None], W[None, :]).sum(2).min(1)[1]
        W_j = W[j]

        f_sg = f.detach()
        W_j_sg = W_j.detach()

        loss = L2_dist(f, W_j_sg).sum(1) + L2_dist(f_sg, W_j).sum(1) * 0.25
        loss = loss.view(-1, int(self.encoder.featureLength /
                                 self.encoder.embedding_dim))
        loss = loss.mean(1)
        return loss

    def vqlizationFeature(self, feature):
        f = feature.view(-1, self.encoder.embedding_dim)
        W = self.encoder.dict.weight

        def L2_dist(a, b):
            return ((a - b) ** 2)

        j = L2_dist(f[:, None], W[None, :]).sum(2).min(1)[1]
        W_j = W[j]

        out = W_j.view(-1, self.encoder.featureLength)

        return out

    def vqlizationWithLoss(self, feature):
        f = feature.view(-1, self.encoder.embedding_dim)
        W = self.encoder.dict.weight

        def L2_dist(a, b):
            return ((a - b) ** 2)

        j = L2_dist(f[:, None], W[None, :]).sum(2).min(1)[1]
        W_j = W[j]

        out = W_j.view(-1, self.encoder.featureLength)

        f_sg = f.detach()
        W_j_sg = W_j.detach()

        loss = L2_dist(f, W_j_sg).sum(1) + L2_dist(f_sg, W_j).sum(1) * 0.25
        loss = loss.view(-1, int(self.encoder.featureLength /
                                 self.encoder.embedding_dim))
        loss = loss.mean(1)

        return loss, out

    def vqlizationWithOutLoss(self, feature):
        f = feature.view(-1, self.encoder.embedding_dim)
        W = self.encoder.dict.weight.detach()

        def L2_dist(a, b):
            return ((a - b) ** 2)

        j = L2_dist(f[:, None], W[None, :]).sum(2).min(1)[1]
        W_j = W[j]

        out = W_j.view(-1, self.encoder.featureLength)

        return out

    def vqlizationWithLoss2(self, feature):
        f = feature.view(-1, self.encoder.embedding_dim)
        W = self.encoder.dict.weight.detach()

        def L2_dist(a, b):
            return ((a - b) ** 2)

        j = L2_dist(f[:, None], W[None, :]).sum(2).min(1)[1]
        W_j = W[j]

        out = W_j.view(-1, self.encoder.featureLength)

        loss = L2_dist(f, W_j).sum(1)
        loss = loss.view(-1, int(self.encoder.featureLength /
                                 self.encoder.embedding_dim))
        loss = loss.mean(1)

        return loss, out
