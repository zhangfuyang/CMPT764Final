import time
import os
import torch
import util
from time import strftime
from datetime import datetime
from dynamicplot import DynamicPlot
from torchfoldext import FoldExt
from chairDataset import ChairDataset, Tree
from torch.autograd import Variable
from modelVQContext import GRASSEncoder, GRASSDecoder, GRASSMerge

config = util.get_args()

config.cuda = not config.no_cuda
if config.gpu < 0 and config.cuda:
    config.gpu = 0
torch.cuda.set_device(config.gpu)
if config.cuda and torch.cuda.is_available():
    print("using CUDA on GPU ", config.gpu)
else:
    print("Not using CUDA")

if config.finetune:
    print("fintune phase")
    encoder = torch.load('./models/vq_encoder_model.pkl')
    decoder = torch.load('./models/vq_decoder_model.pkl')
else:
    encoder = GRASSEncoder(config)
    decoder = GRASSDecoder(config)
if config.cuda:
    encoder.cuda()
    decoder.cuda()
model = GRASSMerge(config, encoder, decoder)

print("Loading data ......", end='', flush=True)
grass_data = ChairDataset(config.data_path)

def my_collate(batch):
    return batch
train_iter = torch.utils.data.DataLoader(grass_data, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
print("DONE")

encoder_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)

print("Start training ...... ")

start = time.time()


if config.save_snapshot:
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    snapshot_folder = os.path.join(config.save_path, 'snapshots_'+strftime("%Y-%m-%d_%H-%M-%S",gmtime()))
    if not os.path.exists(snapshot_folder):
        os.makedirs(snapshot_folder)

if config.save_log:
    fd_log = open('training_log.log', mode='a')
    fd_log.write('\n\nTraining log at '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fd_log.write('\n#epoch: {}'.format(config.epochs))
    fd_log.write('\nbatch_size: {}'.format(config.batch_size))
    fd_log.write('\ncuda: {}'.format(config.cuda))
    fd_log.flush()

header = '    Time    Epoch    Iteration    Progress(%)    Loss'
log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>11.2f}'.split(','))

total_iter = config.epochs * len(train_iter)
if not config.no_plot:
    plot_x = [x for x in range(total_iter)]
    plot_total_loss = [None for x in range(total_iter)]
    dyn_plot = DynamicPlot(title='Training loss over epoches', xdata=plot_x,
                           ydata={'Total_loss':plot_total_loss})
    iter_id = 0
    max_loss = 0

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

def decode_fold(fold, feature, tree):
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

    def decode_node(feature, node):
        if node.is_leaf():
            f = encode_node(node)
            reBox = fold.add('boxNode', feature, f)
            #new_node = Tree.Node(box=reBox, node_type=Tree.NodeType.BOX)
            return fold.add('mseBoxLossLayer', reBox, node.box)

        elif node.is_adj():
            d = fold.add('deSampleLayer', feature)
            if config.finetune:
                vqLoss, _ = fold.add('vqlizationWithLoss2', d).split(2)
                _, feature = fold.add('vqlizationWithLoss2', feature).split(2)
            else:
                vqLoss = fold.add('vqlizationLoss', d)
                feature = fold.add('vqlizationFeature', feature)
            left_node = node.left
            right_node = node.right
            fl = encode_node(left_node)
            fr = encode_node(right_node)
            left_f = fold.add('outterNode', feature, fr)
            right_f = fold.add('outterNode', feature, fl)
            loss1 = decode_node(left_f, node.left)
            loss2 = decode_node(right_f, node.right)
            l_vq = fold.add('vectorAdder', vqLoss, loss1)
            loss = fold.add('vectorAdder',l_vq, loss2)
            return loss
        elif node.is_sym():
            d = fold.add('deSampleLayer', feature)
            if config.finetune:
                vqLoss, _ = fold.add('vqlizationWithLoss2', d).split(2)
                _, feature = fold.add('vqlizationWithLoss2', feature).split(2)
            else:
                vqLoss = fold.add('vqlizationLoss', d)
                feature = fold.add('vqlizationFeature', feature)

            f = encode_node(node)
            new_f, sym_f = fold.add('symParaNode', feature, f).split(2)
            symLoss = fold.add('mseSymLossLayer', sym_f, node.sym)
            loss = decode_node(new_f, node.left)
            l_vq = fold.add('vectorAdder', vqLoss, loss)
            loss = fold.add('vectorAdder', l_vq, symLoss)
            return loss

    loss = decode_node(feature, tree.root)

    return loss


for epoch in range(config.epochs):
    print(header)
    for batch_idx, batch in enumerate(train_iter):
        enc_fold = FoldExt(cuda=config.cuda)
        enc_fold_nodes = []
        for example in batch:
            enc_fold_nodes.append(encode_fold(enc_fold, example))
        enc_fold_nodes = enc_fold.apply(model, [enc_fold_nodes])

        enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)

        dec_fold = FoldExt(cuda=config.cuda)
        dec_fold_nodes = []
        for example, fnode in zip(batch, enc_fold_nodes):
            if config.finetune:
                example.addNoise()
            dec_fold_nodes.append(decode_fold(dec_fold, fnode, example))
        total_loss = dec_fold.apply(model, [dec_fold_nodes])
        total_loss = total_loss[0].sum() / len(batch)
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        total_loss.backward()
        encoder_opt.step()
        decoder_opt.step()
        # report statustucs
        if batch_idx % config.show_log_every == 0:
            print(log_template.format(strftime("%H:%M:%S",time.gmtime(time.time()-start)),
                                      epoch, config.epochs, 1+batch_idx, len(train_iter),
                                      100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.epochs),
                                      total_loss.item()))

        # Plot losses
        if not config.no_plot:
            plot_total_loss[iter_id] = total_loss.data[0]
            max_loss = max(max_loss, total_loss.data[0])
            dyn_plot.setxlim(0., (iter_id+1)*1.05)
            dyn_plot.setylim(0., max_loss*1.05)
            dyn_plot.update_plots(ydata={'Total_loss':plot_total_loss})
            iter_id += 1

    # Save snapshots of the models being trained
    if config.save_snapshot and (epoch + 1) % config.save_snapshot_every == 0:
        print("Saving snapshots of the models ...... ", end='', flush=True)
        torch.save(encoder, snapshot_folder +
                   '//vae_encoder_model_epoch_{}_loss_{:.2f}.pkl'.format(epoch + 1, total_loss.data[0]))
        torch.save(decoder, snapshot_folder +
                   '//vae_decoder_model_epoch_{}_loss_{:.2f}.pkl'.format(epoch + 1, total_loss.data[0]))
        print("DONE")
    # Save training log
    if config.save_log and (epoch + 1) % config.save_log_every == 0:
        fd_log = open('training_log.log', mode='a')
        fd_log.write('\nepoch:{} total_loss:{:.2f}'.format(epoch + 1, total_loss.data[0]))
        fd_log.close()

# Save the final models
print("Saving final models ...... ", end='', flush=True)
if config.finetune:
    torch.save(encoder, config.save_path + '/vq_encoder_model_finetune.pkl')
    torch.save(decoder, config.save_path + '/vq_decoder_model_finetune.pkl')
else:
    torch.save(encoder, config.save_path + '/vq_encoder_model.pkl')
    torch.save(decoder, config.save_path + '/vq_decoder_model.pkl')
print("DONE")

