from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from .mypath_pascal import Path

class VOCSegmentation(Dataset):
    """
    Pascal dataset
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 transform=None
                 ):
        """
        :param base_dir: path to PASCAL dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super(VOCSegmentation).__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationPart')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'list')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line+'.jpg' )
                _cat = os.path.join(self._cat_dir, line +'.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB') # return is RGB pic
        _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return 'PASCAL(split=' + str(self.split) + ')'

class test_segmentation(VOCSegmentation):
    def __init__(self,base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 transform=None,
                 flip=True):
        super(test_segmentation, self).__init__(base_dir=base_dir,split=split,transform=transform)
        self._flip_flag = flip

    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample



