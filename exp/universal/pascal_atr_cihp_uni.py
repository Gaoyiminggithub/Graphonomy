import socket
import timeit
from datetime import datetime
import os
import sys
import glob
import numpy as np
from collections import OrderedDict
sys.path.append('./')
sys.path.append('./networks/')
# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import random

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import pascal, cihp_pascal_atr
from utils import get_iou_from_list
from utils import util as ut
from networks import deeplab_xception_universal, graph
from dataloaders import custom_transforms as tr
from utils import sampler as sam
#
import argparse

'''
source is cihp
target is pascal
'''

gpu_id = 1
# print('Using GPU: {} '.format(gpu_id))

# nEpochs = 100  # Number of epochs for training
resume_epoch = 0   # Default is 0, change if want to resume

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker',default=12,type=int)
    # parser.add_argument('--freezeBN', choices=dict(true=True, false=False), default=True, action=LookupChoices)
    parser.add_argument('--step', default=10, type=int)
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--testepoch', default=10, type=int)
    parser.add_argument('--loadmodel',default='',type=str)
    parser.add_argument('--pretrainedModel', default='', type=str)
    parser.add_argument('--hidden_layers',default=128,type=int)
    parser.add_argument('--gpus',default=4, type=int)
    parser.add_argument('--testInterval', default=5, type=int)
    opts = parser.parse_args()
    return opts

def get_graphs(opts):
    '''source is pascal; target is cihp; middle is atr'''
    # target 1
    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj1_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1 = adj1_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 20, 20).cuda()
    adj1_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20)
    #source 2
    adj2_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj2 = adj2_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 7).cuda()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7)
    # s to target 3
    adj3_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj3 = adj3_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20).transpose(2,3).cuda()
    adj3_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).transpose(2,3)
    # middle 4
    atr_adj = graph.preprocess_adj(graph.atr_graph)
    adj4_ = Variable(torch.from_numpy(atr_adj).float())
    adj4 = adj4_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 18, 18).cuda()
    adj4_test = adj4_.unsqueeze(0).unsqueeze(0).expand(1, 1, 18, 18)
    # source to middle 5
    adj5_ = torch.from_numpy(graph.pascal2atr_nlp_adj).float()
    adj5 = adj5_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 18).cuda()
    adj5_test = adj5_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 18)
    # target to middle 6
    adj6_ = torch.from_numpy(graph.cihp2atr_nlp_adj).float()
    adj6 = adj6_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 20, 18).cuda()
    adj6_test = adj6_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 18)
    train_graph = [adj1, adj2, adj3, adj4, adj5, adj6]
    test_graph = [adj1_test, adj2_test, adj3_test, adj4_test, adj5_test, adj6_test]
    return train_graph, test_graph


def main(opts):
    # Set parameters
    p = OrderedDict()  # Parameters to include in report
    p['trainBatch'] = opts.batch  # Training batch size
    testBatch = 1  # Testing batch size
    useTest = True  # See evolution of the test set when training
    nTestInterval = opts.testInterval # Run on test set every nTestInterval epochs
    snapshot = 1  # Store a model every snapshot epochs
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = opts.lr  # Learning rate
    p['wd'] = 5e-4  # Weight decay
    p['momentum'] = 0.9  # Momentum
    p['epoch_size'] = opts.step  # How many epochs to change learning rate
    p['num_workers'] = opts.numworker
    model_path = opts.pretrainedModel
    backbone = 'xception' # Use xception or resnet as feature extractor
    nEpochs = opts.epochs

    max_id = 0
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    runs = glob.glob(os.path.join(save_dir_root, 'run', 'run_*'))
    for r in runs:
        run_id = int(r.split('_')[-1])
        if run_id >= max_id:
            max_id = run_id + 1
    # run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(max_id))

    # Network definition
    if backbone == 'xception':
        net_ = deeplab_xception_universal.deeplab_xception_end2end_3d(n_classes=20, os=16,
                                                                      hidden_layers=opts.hidden_layers,
                                                                      source_classes=7,
                                                                      middle_classes=18, )
    elif backbone == 'resnet':
        # net_ = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True)
        raise NotImplementedError
    else:
        raise NotImplementedError

    modelName = 'deeplabv3plus-' + backbone + '-voc'+datetime.now().strftime('%b%d_%H-%M-%S')
    criterion = ut.cross_entropy2d

    if gpu_id >= 0:
        # torch.cuda.set_device(device=gpu_id)
        net_.cuda()

    # net load weights
    if not model_path == '':
        x = torch.load(model_path)
        net_.load_state_dict_new(x)
        print('load pretrainedModel.')
    else:
        print('no pretrainedModel.')

    if not opts.loadmodel =='':
        x = torch.load(opts.loadmodel)
        net_.load_source_model(x)
        print('load model:' ,opts.loadmodel)
    else:
        print('no trained model load !!!!!!!!')

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('load model',opts.loadmodel,1)
    writer.add_text('setting',sys.argv[0],1)

    # Use the following optimizer
    optimizer = optim.SGD(net_.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])

    composed_transforms_tr = transforms.Compose([
            tr.RandomSized_new(512),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

    composed_transforms_ts = transforms.Compose([
        tr.Normalize_xception_tf(),
        tr.ToTensor_()])

    composed_transforms_ts_flip = transforms.Compose([
        tr.HorizontalFlip(),
        tr.Normalize_xception_tf(),
        tr.ToTensor_()])

    all_train = cihp_pascal_atr.VOCSegmentation(split='train', transform=composed_transforms_tr, flip=True)
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
    voc_val_flip = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

    num_cihp,num_pascal,num_atr = all_train.get_class_num()
    ss = sam.Sampler_uni(num_cihp,num_pascal,num_atr,opts.batch)
    # balance datasets based pascal
    ss_balanced = sam.Sampler_uni(num_cihp,num_pascal,num_atr,opts.batch, balance_id=1)

    trainloader = DataLoader(all_train, batch_size=p['trainBatch'], shuffle=False, num_workers=p['num_workers'],
                             sampler=ss, drop_last=True)
    trainloader_balanced = DataLoader(all_train, batch_size=p['trainBatch'], shuffle=False, num_workers=p['num_workers'],
                             sampler=ss_balanced, drop_last=True)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=p['num_workers'])
    testloader_flip = DataLoader(voc_val_flip, batch_size=testBatch, shuffle=False, num_workers=p['num_workers'])

    num_img_tr = len(trainloader)
    num_img_balanced = len(trainloader_balanced)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_tr_atr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    print("Training Network")
    net = torch.nn.DataParallel(net_)

    id_list = torch.LongTensor(range(opts.batch))
    pascal_iter = int(num_img_tr//opts.batch)

    # Get graphs
    train_graph, test_graph = get_graphs(opts)
    adj1, adj2, adj3, adj4, adj5, adj6 = train_graph
    adj1_test, adj2_test, adj3_test, adj4_test, adj5_test, adj6_test = test_graph

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, int(1.5*nEpochs)):
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1 and epoch<nEpochs:
            lr_ = ut.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            optimizer = optim.SGD(net_.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
            print('(poly lr policy) learning rate: ', lr_)
            writer.add_scalar('data/lr_',lr_,epoch)
        elif epoch % p['epoch_size'] == p['epoch_size'] - 1 and epoch > nEpochs:
            lr_ = ut.lr_poly(p['lr'], epoch-nEpochs, int(0.5*nEpochs), 0.9)
            optimizer = optim.SGD(net_.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
            print('(poly lr policy) learning rate: ', lr_)
            writer.add_scalar('data/lr_', lr_, epoch)

        net_.train()
        if epoch < nEpochs:
            for ii, sample_batched in enumerate(trainloader):
                inputs, labels = sample_batched['image'], sample_batched['label']
                dataset_lbl = sample_batched['pascal'][0].item()
                # Forward-Backward of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                global_step += 1

                if gpu_id >= 0:
                    inputs, labels = inputs.cuda(), labels.cuda()

                if dataset_lbl == 0:
                    # 0 is cihp -- target
                    _, outputs,_ = net.forward(None, input_target=inputs, input_middle=None, adj1_target=adj1, adj2_source=adj2,
                        adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2,3), adj4_middle=adj4,adj5_transfer_s2m=adj5.transpose(2, 3),
                        adj6_transfer_t2m=adj6.transpose(2, 3),adj5_transfer_m2s=adj5,adj6_transfer_m2t=adj6,)
                elif dataset_lbl == 1:
                    # pascal is source
                    outputs, _, _ = net.forward(inputs, input_target=None, input_middle=None, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                else:
                    # atr
                    _, _, outputs = net.forward(None, input_target=None, input_middle=inputs, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                # print(sample_batched['pascal'])
                # print(outputs.size(),)
                # print(labels)
                loss = criterion(outputs, labels,  batch_average=True)
                running_loss_tr += loss.item()

                # Print stuff
                if ii % num_img_tr == (num_img_tr - 1):
                    running_loss_tr = running_loss_tr / num_img_tr
                    writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                    print('[Epoch: %d, numImages: %5d]' % (epoch, epoch))
                    print('Loss: %f' % running_loss_tr)
                    running_loss_tr = 0
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")

                # Backward the averaged gradient
                loss /= p['nAveGrad']
                loss.backward()
                aveGrad += 1

                # Update the weights once in p['nAveGrad'] forward passes
                if aveGrad % p['nAveGrad'] == 0:
                    writer.add_scalar('data/total_loss_iter', loss.item(), global_step)
                    if dataset_lbl == 0:
                        writer.add_scalar('data/total_loss_iter_cihp', loss.item(), global_step)
                    if dataset_lbl == 1:
                        writer.add_scalar('data/total_loss_iter_pascal', loss.item(), global_step)
                    if dataset_lbl == 2:
                        writer.add_scalar('data/total_loss_iter_atr', loss.item(), global_step)
                    optimizer.step()
                    optimizer.zero_grad()
                    # optimizer_gcn.step()
                    # optimizer_gcn.zero_grad()
                    aveGrad = 0

                # Show 10 * 3 images results each epoch
                if ii % (num_img_tr // 10) == 0:
                    grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                    writer.add_image('Image', grid_image, global_step)
                    grid_image = make_grid(ut.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
                                           range=(0, 255))
                    writer.add_image('Predicted label', grid_image, global_step)
                    grid_image = make_grid(ut.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
                    writer.add_image('Groundtruth label', grid_image, global_step)

                print('loss is ',loss.cpu().item(),flush=True)
        else:
            # Balanced the number of datasets
            for ii, sample_batched in enumerate(trainloader_balanced):
                inputs, labels = sample_batched['image'], sample_batched['label']
                dataset_lbl = sample_batched['pascal'][0].item()
                # Forward-Backward of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                global_step += 1

                if gpu_id >= 0:
                    inputs, labels = inputs.cuda(), labels.cuda()

                if dataset_lbl == 0:
                    # 0 is cihp -- target
                    _, outputs, _ = net.forward(None, input_target=inputs, input_middle=None, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                elif dataset_lbl == 1:
                    # pascal is source
                    outputs, _, _ = net.forward(inputs, input_target=None, input_middle=None, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                else:
                    # atr
                    _, _, outputs = net.forward(None, input_target=None, input_middle=inputs, adj1_target=adj1,
                                                adj2_source=adj2,
                                                adj3_transfer_s2t=adj3, adj3_transfer_t2s=adj3.transpose(2, 3),
                                                adj4_middle=adj4, adj5_transfer_s2m=adj5.transpose(2, 3),
                                                adj6_transfer_t2m=adj6.transpose(2, 3), adj5_transfer_m2s=adj5,
                                                adj6_transfer_m2t=adj6, )
                # print(sample_batched['pascal'])
                # print(outputs.size(),)
                # print(labels)
                loss = criterion(outputs, labels, batch_average=True)
                running_loss_tr += loss.item()

                # Print stuff
                if ii % num_img_balanced == (num_img_balanced - 1):
                    running_loss_tr = running_loss_tr / num_img_balanced
                    writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                    print('[Epoch: %d, numImages: %5d]' % (epoch, epoch))
                    print('Loss: %f' % running_loss_tr)
                    running_loss_tr = 0
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")

                # Backward the averaged gradient
                loss /= p['nAveGrad']
                loss.backward()
                aveGrad += 1

                # Update the weights once in p['nAveGrad'] forward passes
                if aveGrad % p['nAveGrad'] == 0:
                    writer.add_scalar('data/total_loss_iter', loss.item(), global_step)
                    if dataset_lbl == 0:
                        writer.add_scalar('data/total_loss_iter_cihp', loss.item(), global_step)
                    if dataset_lbl == 1:
                        writer.add_scalar('data/total_loss_iter_pascal', loss.item(), global_step)
                    if dataset_lbl == 2:
                        writer.add_scalar('data/total_loss_iter_atr', loss.item(), global_step)
                    optimizer.step()
                    optimizer.zero_grad()

                    aveGrad = 0

                # Show 10 * 3 images results each epoch
                if ii % (num_img_balanced // 10) == 0:
                    grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                    writer.add_image('Image', grid_image, global_step)
                    grid_image = make_grid(
                        ut.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3,
                        normalize=False,
                        range=(0, 255))
                    writer.add_image('Predicted label', grid_image, global_step)
                    grid_image = make_grid(
                        ut.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3,
                        normalize=False, range=(0, 255))
                    writer.add_image('Groundtruth label', grid_image, global_step)

                print('loss is ', loss.cpu().item(), flush=True)

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net_.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth')))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            val_pascal(net_=net_, testloader=testloader, testloader_flip=testloader_flip, test_graph=test_graph,
                       criterion=criterion, epoch=epoch, writer=writer)


def val_pascal(net_, testloader, testloader_flip, test_graph, criterion, epoch, writer, classes=7):
    running_loss_ts = 0.0
    miou = 0
    adj1_test, adj2_test, adj3_test, adj4_test, adj5_test, adj6_test = test_graph
    num_img_ts = len(testloader)
    net_.eval()
    pred_list = []
    label_list = []
    for ii, sample_batched in enumerate(zip(testloader, testloader_flip)):
        # print(ii)
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = torch.cat((inputs, inputs_f), dim=0)
        # Forward pass of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)

        with torch.no_grad():
            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs, _, _ = net_.forward(inputs, input_target=None, input_middle=None,
                                         adj1_target=adj1_test.cuda(),
                                         adj2_source=adj2_test.cuda(),
                                         adj3_transfer_s2t=adj3_test.cuda(),
                                         adj3_transfer_t2s=adj3_test.transpose(2, 3).cuda(),
                                         adj4_middle=adj4_test.cuda(),
                                         adj5_transfer_s2m=adj5_test.transpose(2, 3).cuda(),
                                         adj6_transfer_t2m=adj6_test.transpose(2, 3).cuda(),
                                         adj5_transfer_m2s=adj5_test.cuda(),
                                         adj6_transfer_m2t=adj6_test.cuda(), )
        # pdb.set_trace()
        outputs = (outputs[0] + flip(outputs[1], dim=-1)) / 2
        outputs = outputs.unsqueeze(0)
        predictions = torch.max(outputs, 1)[1]
        pred_list.append(predictions.cpu())
        label_list.append(labels.squeeze(1).cpu())
        loss = criterion(outputs, labels, batch_average=True)
        running_loss_ts += loss.item()

        # total_iou += utils.get_iou(predictions, labels)

        # Print stuff
        if ii % num_img_ts == num_img_ts - 1:
            # if ii == 10:
            miou = get_iou_from_list(pred_list, label_list, n_cls=classes)
            running_loss_ts = running_loss_ts / num_img_ts

            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii * 1 + inputs.data.shape[0]))
            writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
            writer.add_scalar('data/test_miour', miou, epoch)
            print('Loss: %f' % running_loss_ts)
            print('MIoU: %f\n' % miou)
    # return miou


if __name__ == '__main__':
    opts = get_parser()
    main(opts)