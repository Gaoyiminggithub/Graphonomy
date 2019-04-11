from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .mypath_cihp import Path
from .mypath_pascal import Path as PP
from .mypath_atr import Path as PA
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VOCSegmentation(Dataset):
    """
    Pascal dataset
    """

    def __init__(self,
                 cihp_dir=Path.db_root_dir('cihp'),
                 split='train',
                 transform=None,
                 flip=False,
                 pascal_dir = PP.db_root_dir('pascal'),
                 atr_dir = PA.db_root_dir('atr'),
                 ):
        """
        :param cihp_dir: path to CIHP dataset directory
        :param pascal_dir: path to PASCAL dataset directory
        :param atr_dir: path to ATR dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super(VOCSegmentation).__init__()
        ## for cihp
        self._flip_flag = flip
        self._base_dir = cihp_dir
        self._image_dir = os.path.join(self._base_dir, 'Images')
        self._cat_dir = os.path.join(self._base_dir, 'Category_ids')
        self._flip_dir = os.path.join(self._base_dir,'Category_rev_ids')
        ## for Pascal
        self._base_dir_pascal = pascal_dir
        self._image_dir_pascal = os.path.join(self._base_dir_pascal, 'JPEGImages')
        self._cat_dir_pascal = os.path.join(self._base_dir_pascal, 'SegmentationPart')
        # self._flip_dir_pascal = os.path.join(self._base_dir_pascal, 'Category_rev_ids')
        ## for atr
        self._base_dir_atr = atr_dir
        self._image_dir_atr = os.path.join(self._base_dir_atr, 'JPEGImages')
        self._cat_dir_atr = os.path.join(self._base_dir_atr, 'SegmentationClassAug')
        self._flip_dir_atr = os.path.join(self._base_dir_atr, 'SegmentationClassAug_rev')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'lists')
        _splits_dir_pascal = os.path.join(self._base_dir_pascal, 'list')
        _splits_dir_atr = os.path.join(self._base_dir_atr, 'list')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.flip_categories = []
        self.datasets_lbl = []

        # num
        self.num_cihp = 0
        self.num_pascal = 0
        self.num_atr = 0
        # for cihp is 0
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()
            self.num_cihp += len(lines)
            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line+'.jpg' )
                _cat = os.path.join(self._cat_dir, line +'.png')
                _flip = os.path.join(self._flip_dir,line + '.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append(_flip)
                self.datasets_lbl.append(0)

        # for pascal is 1
        for splt in self.split:
            if splt == 'test':
                splt='val'
            with open(os.path.join(os.path.join(_splits_dir_pascal, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()
            self.num_pascal += len(lines)
            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir_pascal, line+'.jpg' )
                _cat = os.path.join(self._cat_dir_pascal, line +'.png')
                # _flip = os.path.join(self._flip_dir,line + '.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                # assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append([])
                self.datasets_lbl.append(1)

        # for atr is 2
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir_atr, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()
            self.num_atr += len(lines)
            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir_atr, line + '.jpg')
                _cat = os.path.join(self._cat_dir_atr, line + '.png')
                _flip = os.path.join(self._flip_dir_atr, line + '.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append(_flip)
                self.datasets_lbl.append(2)

        assert (len(self.images) == len(self.categories))
        # assert len(self.flip_categories) == len(self.categories)

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def get_class_num(self):
        return self.num_cihp,self.num_pascal,self.num_atr



    def __getitem__(self, index):
        _img, _target,_lbl= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target,}

        if self.transform is not None:
            sample = self.transform(sample)
        sample['pascal'] = _lbl
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')  # return is RGB pic
        type_lbl = self.datasets_lbl[index]
        if self._flip_flag:
            if random.random() < 0.5 :
                # _target = Image.open(self.flip_categories[index])
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                if type_lbl == 0 or type_lbl == 2:
                    _target = Image.open(self.flip_categories[index])
                else:
                    _target = Image.open(self.categories[index])
                    _target = _target.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                _target = Image.open(self.categories[index])
        else:
            _target = Image.open(self.categories[index])

        return _img, _target,type_lbl

    def __str__(self):
        return 'datasets(split=' + str(self.split) + ')'












if __name__ == '__main__':
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        # tr.RandomHorizontalFlip(),
        tr.RandomSized_new(512),
        tr.RandomRotate(15),
        tr.ToTensor_()])



    voc_train = VOCSegmentation(split='train',
                                transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=1)

    for ii, sample in enumerate(dataloader):
        if ii >10:
            break