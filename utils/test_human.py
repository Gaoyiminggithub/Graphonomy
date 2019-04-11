import os
import numpy as np
from PIL import Image


def main():
    image_paths, label_paths = init_path()
    hist = compute_hist(image_paths, label_paths)
    show_result(hist)


def init_path():
    list_file = './human/list/val_id.txt'
    file_names = []
    with open(list_file, 'rb') as f:
        for fn in f:
            file_names.append(fn.strip())

    image_dir = './human/features/attention/val/results/'
    label_dir = './human/data/labels/'

    image_paths = []
    label_paths = []
    for file_name in file_names:
        image_paths.append(os.path.join(image_dir, file_name + '.png'))
        label_paths.append(os.path.join(label_dir, file_name + '.png'))
    return image_paths, label_paths


def fast_hist(lbl, pred, n_cls):
    '''
    compute the miou
    :param lbl: label
    :param pred: output
    :param n_cls: num of class
    :return:
    '''
    # print(n_cls)
    k = (lbl >= 0) & (lbl < n_cls)
    return np.bincount(n_cls * lbl[k].astype(int) + pred[k], minlength=n_cls ** 2).reshape(n_cls, n_cls)


def compute_hist(images, labels,n_cls=20):
    hist = np.zeros((n_cls, n_cls))
    for img_path, label_path in zip(images, labels):
        label = Image.open(label_path)
        label_array = np.array(label, dtype=np.int32)
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.int32)

        gtsz = label_array.shape
        imgsz = image_array.shape
        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.int32)

        hist += fast_hist(label_array, image_array, n_cls)

    return hist


def show_result(hist):
    classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
               'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
               'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
               'rightShoe']
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print('=' * 50)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    print('-' * 50)

    # @evaluation 2: mean accuracy & per-class accuracy
    print('Accuracy for each class (pixel accuracy):')
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    print('>>>', 'mean accuracy', np.nanmean(acc))
    print('-' * 50)

    # @evaluation 3: mean IU & per-class IU
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IU', np.nanmean(iu))
    print('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print('>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum())
    print('=' * 50)

def get_iou(pred,lbl,n_cls):
    '''
    need tensor cpu
    :param pred:
    :param lbl:
    :param n_cls:
    :return:
    '''
    hist = np.zeros((n_cls,n_cls))
    for i,j in zip(range(pred.size(0)),range(lbl.size(0))):
        pred_item = pred[i].data.numpy()
        lbl_item = lbl[j].data.numpy()
        hist += fast_hist(lbl_item, pred_item, n_cls)
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    # for i in range(20):
    #     print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IU', np.nanmean(iu))
    miou = np.nanmean(iu)
    print('-' * 50)
    return miou

def get_iou_from_list(pred,lbl,n_cls):
    '''
    need tensor cpu
    :param pred: list
    :param lbl: list
    :param n_cls:
    :return:
    '''
    hist = np.zeros((n_cls,n_cls))
    for i,j in zip(range(len(pred)),range(len(lbl))):
        pred_item = pred[i].data.numpy()
        lbl_item = lbl[j].data.numpy()
        # print(pred_item.shape,lbl_item.shape)
        hist += fast_hist(lbl_item, pred_item, n_cls)

    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    # for i in range(20):
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    print('-' * 50)
    #     print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IU', np.nanmean(iu))
    miou = np.nanmean(iu)
    print('-' * 50)

    acc = num_cor_pix / num_gt_pix
    print('>>>', 'mean accuracy', np.nanmean(acc))
    print('-' * 50)

    return miou


if __name__ == '__main__':
    import torch
    pred = torch.autograd.Variable(torch.ones((2,1,32,32)).int())*20
    pred2 = torch.autograd.Variable(torch.zeros((2,1, 32, 32)).int())
    # lbl = [torch.zeros((32,32)).int() for _ in range(len(pred))]
    get_iou(pred,pred2,7)
