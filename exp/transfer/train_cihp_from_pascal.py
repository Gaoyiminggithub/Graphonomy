import socket
import timeit
from datetime import datetime
import os
import sys
import glob
import numpy as np
from collections import OrderedDict
sys.path.append('../../')
sys.path.append('../../networks/')
# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import cihp
from utils import util,get_iou_from_list
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#
import argparse

gpu_id = 0

nEpochs = 100  # Number of epochs for training
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
    parser.add_argument('--freezeBN', choices=dict(true=True, false=False), default=True, action=LookupChoices)
    parser.add_argument('--step', default=10, type=int)
    parser.add_argument('--classes', default=20, type=int)
    parser.add_argument('--testInterval', default=10, type=int)
    parser.add_argument('--loadmodel',default='',type=str)
    parser.add_argument('--pretrainedModel', default='', type=str)
    parser.add_argument('--hidden_layers',default=128,type=int)
    parser.add_argument('--gpus',default=4, type=int)

    opts = parser.parse_args()
    return opts

def get_graphs(opts):
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2 = adj2_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20).transpose(2, 3).cuda()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3 = adj1_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 7).cuda()
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7)

    # adj2 = torch.from_numpy(graph.cihp2pascal_adj).float()
    # adj2 = adj2.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20)
    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1 = adj3_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 20, 20).cuda()
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20)
    train_graph = [adj1, adj2, adj3]
    test_graph = [adj1_test, adj2_test, adj3_test]
    return train_graph, test_graph


def val_cihp(net_, testloader, testloader_flip, test_graph, epoch, writer, criterion, classes=20):
    adj1_test, adj2_test, adj3_test = test_graph
    num_img_ts = len(testloader)
    net_.eval()
    pred_list = []
    label_list = []
    running_loss_ts = 0.0
    miou = 0
    for ii, sample_batched in enumerate(zip(testloader, testloader_flip)):

        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = torch.cat((inputs, inputs_f), dim=0)
        # Forward pass of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)
        if gpu_id >= 0:
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = net_.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
        # pdb.set_trace()
        outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
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


def main(opts):
    p = OrderedDict()  # Parameters to include in report
    p['trainBatch'] = opts.batch  # Training batch size
    testBatch = 1  # Testing batch size
    useTest = True  # See evolution of the test set when training
    nTestInterval = opts.testInterval # Run on test set every nTestInterval epochs
    snapshot = 1  # Store a model every snapshot epochs
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = opts.lr  # Learning rate
    p['lrFtr'] = 1e-5
    p['lraspp'] = 1e-5
    p['lrpro'] = 1e-5
    p['lrdecoder'] = 1e-5
    p['lrother']  = 1e-5
    p['wd'] = 5e-4  # Weight decay
    p['momentum'] = 0.9  # Momentum
    p['epoch_size'] = opts.step  # How many epochs to change learning rate
    p['num_workers'] = opts.numworker
    model_path = opts.pretrainedModel
    backbone = 'xception' # Use xception or resnet as feature extractor,
    nEpochs = opts.epochs

    max_id = 0
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    runs = glob.glob(os.path.join(save_dir_root, 'run_cihp', 'run_*'))
    for r in runs:
        run_id = int(r.split('_')[-1])
        if run_id >= max_id:
            max_id = run_id + 1
    save_dir = os.path.join(save_dir_root, 'run_cihp', 'run_' + str(max_id))

    # Network definition
    if backbone == 'xception':
        net_ = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=opts.classes, os=16,
                                                                                      hidden_layers=opts.hidden_layers, source_classes=7, )
    elif backbone == 'resnet':
        # net_ = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True)
        raise NotImplementedError
    else:
        raise NotImplementedError

    modelName = 'deeplabv3plus-' + backbone + '-voc'+datetime.now().strftime('%b%d_%H-%M-%S')
    criterion = util.cross_entropy2d

    if gpu_id >= 0:
        # torch.cuda.set_device(device=gpu_id)
        net_.cuda()

    # net load weights
    if not model_path == '':
        x = torch.load(model_path)
        net_.load_state_dict_new(x)
        print('load pretrainedModel:', model_path)
    else:
        print('no pretrainedModel.')
    if not opts.loadmodel =='':
        x = torch.load(opts.loadmodel)
        net_.load_source_model(x)
        print('load model:' ,opts.loadmodel)
    else:
        print('no model load !!!!!!!!')

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('load model',opts.loadmodel,1)
    writer.add_text('setting',sys.argv[0],1)

    if opts.freezeBN:
        net_.freeze_bn()

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

    voc_train = cihp.VOCSegmentation(split='train', transform=composed_transforms_tr, flip=True)
    voc_val = cihp.VOCSegmentation(split='val', transform=composed_transforms_ts)
    voc_val_flip = cihp.VOCSegmentation(split='val', transform=composed_transforms_ts_flip)

    trainloader = DataLoader(voc_train, batch_size=p['trainBatch'], shuffle=True, num_workers=p['num_workers'],drop_last=True)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=p['num_workers'])
    testloader_flip = DataLoader(voc_val_flip, batch_size=testBatch, shuffle=False, num_workers=p['num_workers'])

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    print("Training Network")

    net = torch.nn.DataParallel(net_)
    train_graph, test_graph = get_graphs(opts)
    adj1, adj2, adj3 = train_graph


    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1:
            lr_ = util.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            optimizer = optim.SGD(net_.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
            writer.add_scalar('data/lr_', lr_, epoch)
            print('(poly lr policy) learning rate: ', lr_)

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, labels = sample_batched['image'], sample_batched['label']
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]

            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs, adj1, adj3, adj2)

            loss = criterion(outputs, labels, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
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
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            # Show 10 * 3 images results each epoch
            if ii % (num_img_tr // 10) == 0:
                grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Image', grid_image, global_step)
                grid_image = make_grid(util.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
                                       range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(util.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)
            print('loss is ', loss.cpu().item(), flush=True)

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net_.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth')))

        torch.cuda.empty_cache()

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            val_cihp(net_,testloader=testloader, testloader_flip=testloader_flip, test_graph=test_graph,
                     epoch=epoch,writer=writer,criterion=criterion, classes=opts.classes)
        torch.cuda.empty_cache()




if __name__ == '__main__':
    opts = get_parser()
    main(opts)