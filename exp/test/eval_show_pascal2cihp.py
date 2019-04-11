import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
import glob
from collections import OrderedDict
sys.path.append('../../')
# PyTorch includes
import torch
import pdb
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import cv2

# Tensorboard include
# from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import  cihp
from utils import util
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#
import argparse
import copy
import torch.nn.functional as F
from test_from_disk import eval_


gpu_id = 1

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


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

def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker', default=12, type=int)
    parser.add_argument('--step', default=30, type=int)
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--testepoch', default=10, type=int)
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--txt_file', default='', type=str)
    parser.add_argument('--hidden_layers', default=128, type=int)
    parser.add_argument('--gpus', default=4, type=int)
    parser.add_argument('--output_path', default='./results/', type=str)
    parser.add_argument('--gt_path', default='./results/', type=str)
    opts = parser.parse_args()
    return opts


def main(opts):
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    p = OrderedDict()  # Parameters to include in report
    p['trainBatch'] = opts.batch  # Training batch size
    p['nAveGrad'] = 1  # Average the gradient of several iterations
    p['lr'] = opts.lr  # Learning rate
    p['lrFtr'] = 1e-5
    p['lraspp'] = 1e-5
    p['lrpro'] = 1e-5
    p['lrdecoder'] = 1e-5
    p['lrother']  = 1e-5
    p['wd'] = 5e-4  # Weight decay
    p['momentum'] = 0.9  # Momentum
    p['epoch_size'] = 10  # How many epochs to change learning rate
    p['num_workers'] = opts.numworker
    backbone = 'xception' # Use xception or resnet as feature extractor,

    with open(opts.txt_file, 'r') as f:
        img_list = f.readlines()

    max_id = 0
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    runs = glob.glob(os.path.join(save_dir_root, 'run', 'run_*'))
    for r in runs:
        run_id = int(r.split('_')[-1])
        if run_id >= max_id:
            max_id = run_id + 1
    # run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    # Network definition
    if backbone == 'xception':
        net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=opts.classes, os=16,
                                                                                     hidden_layers=opts.hidden_layers, source_classes=7,
                                                                                     )
    elif backbone == 'resnet':
        # net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True)
        raise NotImplementedError
    else:
        raise NotImplementedError

    if gpu_id >= 0:
        net.cuda()

    # net load weights
    if not opts.loadmodel =='':
        x = torch.load(opts.loadmodel)
        net.load_source_model(x)
        print('load model:' ,opts.loadmodel)
    else:
        print('no model load !!!!!!!!')

    ## multi scale
    scale_list=[1,0.5,0.75,1.25,1.5,1.75]
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_(pv),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_(pv),
            tr.HorizontalFlip(),
            tr.Normalize_xception_tf(),
            tr.ToTensor_()])

        voc_val = cihp.VOCSegmentation(split='test', transform=composed_transforms_ts)
        voc_val_f = cihp.VOCSegmentation(split='test', transform=composed_transforms_ts_flip)

        testloader = DataLoader(voc_val, batch_size=1, shuffle=False, num_workers=p['num_workers'])
        testloader_flip = DataLoader(voc_val_f, batch_size=1, shuffle=False, num_workers=p['num_workers'])

        testloader_list.append(copy.deepcopy(testloader))
        testloader_flip_list.append(copy.deepcopy(testloader_flip))

    print("Eval Network")

    if not os.path.exists(opts.output_path + 'cihp_output_vis/'):
        os.makedirs(opts.output_path + 'cihp_output_vis/')
    if not os.path.exists(opts.output_path + 'cihp_output/'):
        os.makedirs(opts.output_path + 'cihp_output/')

    start_time = timeit.default_timer()
    # One testing epoch
    total_iou = 0.0
    net.eval()
    for ii, large_sample_batched in enumerate(zip(*testloader_list, *testloader_flip_list)):
        print(ii)
        #1 0.5 0.75 1.25 1.5 1.75 ; flip:
        sample1 = large_sample_batched[:6]
        sample2 = large_sample_batched[6:]
        for iii,sample_batched in enumerate(zip(sample1,sample2)):
            inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
            inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
            inputs = torch.cat((inputs,inputs_f),dim=0)
            if iii == 0:
                _,_,h,w = inputs.size()
            # assert inputs.size() == inputs_f.size()

            # Forward pass of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=False), Variable(labels)

            with torch.no_grad():
                if gpu_id >= 0:
                    inputs, labels = inputs.cuda(), labels.cuda()
                # outputs = net.forward(inputs)
                # pdb.set_trace()
                outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
                outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
                outputs = outputs.unsqueeze(0)

                if iii>0:
                    outputs = F.upsample(outputs,size=(h,w),mode='bilinear',align_corners=True)
                    outputs_final = outputs_final + outputs
                else:
                    outputs_final = outputs.clone()
        ################ plot pic
        predictions = torch.max(outputs_final, 1)[1]
        prob_predictions = torch.max(outputs_final,1)[0]
        results = predictions.cpu().numpy()
        prob_results = prob_predictions.cpu().numpy()
        vis_res = decode_labels(results)

        parsing_im = Image.fromarray(vis_res[0])
        parsing_im.save(opts.output_path + 'cihp_output_vis/{}.png'.format(img_list[ii][:-1]))
        cv2.imwrite(opts.output_path + 'cihp_output/{}.png'.format(img_list[ii][:-1]), results[0,:,:])
        # np.save('../../cihp_prob_output/{}.npy'.format(img_list[ii][:-1]), prob_results[0, :, :])
        # pred_list.append(predictions.cpu())
        # label_list.append(labels.squeeze(1).cpu())
        # loss = criterion(outputs, labels, batch_average=True)
        # running_loss_ts += loss.item()

        # total_iou += utils.get_iou(predictions, labels)
    end_time = timeit.default_timer()
    print('time use for '+str(ii) + ' is :' + str(end_time - start_time))

    # Eval
    pred_path = opts.output_path + 'cihp_output/'
    eval_(pred_path=pred_path, gt_path=opts.gt_path,classes=opts.classes, txt_file=opts.txt_file)

if __name__ == '__main__':
    opts = get_parser()
    main(opts)