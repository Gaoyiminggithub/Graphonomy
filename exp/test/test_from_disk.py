import sys
sys.path.append('./')
# PyTorch includes
import torch
import numpy as np

from utils import test_human
from PIL import Image

#
import argparse

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker',default=12,type=int)
    parser.add_argument('--freezeBN', choices=dict(true=True, false=False), default=True, action=LookupChoices)
    parser.add_argument('--step', default=30, type=int)
    parser.add_argument('--txt_file',default=None,type=str)
    parser.add_argument('--pred_path',default=None,type=str)
    parser.add_argument('--gt_path',default=None,type=str)
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--testepoch', default=10, type=int)
    opts = parser.parse_args()
    return opts

def eval_(pred_path, gt_path, classes, txt_file):
    pred_path = pred_path
    gt_path = gt_path

    with open(txt_file,) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    output_list = []
    label_list = []
    for i,file in enumerate(lines):
        print(i)
        file_name = file + '.png'
        try:
            predict_pic = np.array(Image.open(pred_path+file_name))
            gt_pic = np.array(Image.open(gt_path+file_name))
            output_list.append(torch.from_numpy(predict_pic))
            label_list.append(torch.from_numpy(gt_pic))
        except:
            print(file_name,flush=True)
            raise RuntimeError('no predict/gt image.')
            # gt_pic = np.array(Image.open(gt_path + file_name))
            # output_list.append(torch.from_numpy(gt_pic))
            # label_list.append(torch.from_numpy(gt_pic))


    miou = test_human.get_iou_from_list(output_list, label_list, n_cls=classes)

    print('Validation:')
    print('MIoU: %f\n' % miou)

if __name__ == '__main__':
    opts = get_parser()
    eval_(pred_path=opts.pred_path, gt_path=opts.gt_path, classes=opts.classes, txt_file=opts.txt_file)