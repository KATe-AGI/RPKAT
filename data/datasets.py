'''
Build trainining/testing datasets
'''
import os
import torch
import random

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

def build_dataset(is_train, args):
    # Apply transformations based on whether the dataset is for training or testing
    transform = build_transform(is_train, args)

    # These datasets are pre-adjusted to ImageNet format
    if args.data_set == 'HFUT-VL1':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 80
    if args.data_set == 'HFUT-VL2':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 80
    elif args.data_set == 'CompCars':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 281
    elif args.data_set == 'CompCars_L':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 68
    elif args.data_set == 'CompCars_C':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 11
    elif args.data_set == 'Frontal-103':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1759
    elif args.data_set == 'UFPR-VCR':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 11
    elif args.data_set == 'SBOD':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 162
    elif args.data_set == 'SBOD_C':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 8
    
    #!
    # Apply the scaling factor to the training set
    if is_train and args.scale_factor < 1.0:
        random.seed(args.seed)  # Set the seed for reproducibility
        torch.manual_seed(args.seed)
        num_samples = int(len(dataset) * args.scale_factor)
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    
    if args.finetune:
        t.append(
            transforms.Resize((args.input_size, args.input_size),
                              interpolation=InterpolationMode.BICUBIC)
            )
    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
