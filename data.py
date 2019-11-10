import os
import os.path as osp
import cv2

import torch
import numpy as np
from PIL import Image
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, iterable_to_str
import torchvision
from torch.utils import data

import collections


class Cityscapes(VisionDataset):

    # Based on https://github.com/mcordts/cityscapesScripts
    # CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
    #                                                  'has_instances', 'ignore_in_eval', 'color'])
    #
    # classes = [
    #     CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    #     CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    #     CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    #     CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    #     CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    #     CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    #     CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    #     CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    #     CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    #     CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    #     CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    #     CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    #     CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    #     CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    #     CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    #     CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    #     CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    #     CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    #     CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    #     CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    #     CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    #     CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    #     CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    #     CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    #     CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    #     CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    #     CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    #     CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    #     CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    #     CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    # ]

    def __init__(self, root,
                    split='train',
                    mode='fine',
                    target_type='instance',
                    transforms = None):


        super(Cityscapes, self).__init__(root, transforms)

        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color"))
         for value in self.target_type]


        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))

            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)


    def do_transform(self, image, mask):

        resize_shp = self.transforms['resize']
        resize = torchvision.transforms.Resize(resize_shp)

        image = resize(image)
        mask = resize(mask)
        # Random crop
        if self.split == 'train':

            random_crop_size = self.transforms['random_crop']
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                image, output_size=random_crop_size)
            image = torchvision.transforms.functional.crop(image, i, j, h, w)
            mask = torchvision.transforms.functional.crop(mask, i, j, h, w)

        return image, mask

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i]).convert('P')

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:

            # image = self.transforms(image)
            # target = self.transforms(target)


            image, target = self.do_transform(image, target)

            to_tensor = torchvision.transforms.ToTensor()
            normalize = torchvision.transforms.Normalize(
                                mean = [.5,.5,.5],
                                std = [.5,.5,.5])
            image = normalize(to_tensor(image))
            target = torch.LongTensor(np.asarray(target))


        return image, target, index


    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

class Synthia(Dataset):

    def __init__(self, root, crop_size,
                random_crop,
                 mean=(128, 128, 128),
                 scale=True,
                 mirror=True,
                 ignore_label=255,
                  list_path = None,
                  max_iters=None):

        super(Synthia, self).__init__()

        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror

        # if not max_iters==None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        # self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        #                       19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
        #                       26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in os.listdir(osp.join(self.root, "RGB/")):
            #
            id = name.split('/')[-1]

            img_file = osp.join(self.root, "RGB/%s" % id)
            label_file = osp.join(self.root, "GT/LABELS16/%s" % id)
            depth_im = osp.join(self.root, "Depth/Depth/%s" % id)
            color_file = osp.join(self.root, "GT/COLOR/%s" % id)

            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "depth": depth_im,
                "color": color_file
            })

    def __len__(self):
        return len(self.files)

    def do_transform(self, image, mask, depth):

        resize_shp = self.crop_size
        resize = torchvision.transforms.Resize(resize_shp)
        image = resize(image)
        mask = resize(mask)
        depth = resize(depth)

        # Random crop
        random_crop_size = self.random_crop
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            image, output_size=random_crop_size)

        image = torchvision.transforms.functional.crop(image, i, j, h, w)
        mask = torchvision.transforms.functional.crop(mask, i, j, h, w)
        depth = torchvision.transforms.functional.crop(depth, i, j, h, w)

        return image, mask, depth

    def __getitem__(self, index):


        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"]).convert('P')
        depth_im = Image.open(datafiles['depth']).convert('P')
        color_im = Image.open(datafiles['color'])

        name = datafiles["name"]

        image, labels, depth = self.do_transform(image, label, depth_im)

        to_tensor = torchvision.transforms.ToTensor()
        normalize = torchvision.transforms.Normalize(
                            mean = [.5,.5,.5],
                            std = [.5,.5,.5])
        image = normalize(to_tensor(image))
        labels = torch.LongTensor(np.asarray(labels))
        depth = to_tensor(depth)

        return image, labels, depth, index
