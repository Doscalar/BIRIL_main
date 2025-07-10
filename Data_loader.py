import datetime
import os
import time
import warnings
from spikingjelly.activation_based.model.tv_ref_classify import presets, transforms, utils
import torch.utils.data
import torchvision
from spikingjelly.activation_based.model.tv_ref_classify.sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
import logging
from tqdm import tqdm
from spikingjelly.activation_based import functional
import torch
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet, train_classify
import train
try:
    from torchvision import prototype
except ImportError:
    prototype = None


def load_CIFAR10( args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        ),
    )

    print("Took", time.time() - st)

    print("Loading validation data")

    dataset_test = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    )

    print("Creating data loaders")
    loader_g = torch.Generator()
    loader_g.manual_seed(args.seed)

    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def load_CIFAR100(args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.CIFAR100(
        root=args.data_path,
        train=True,
        download=True,
        transform=presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        ),
    )

    print("Took", time.time() - st)

    print("Loading validation data")

    dataset_test = torchvision.datasets.CIFAR100(
        root=args.data_path,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]),
    )

    print("Creating data loaders")
    loader_g = torch.Generator()
    loader_g.manual_seed(args.seed)

    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def load_MINIST( args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)
    print("Loading training data")
    st = time.time()
    dataset = torchvision.datasets.MNIST(
        root=args.data_path,
        train=True,
        download=True,
        transform=presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            mean=(0.1307,),
            std=(0.3081,),
            interpolation=interpolation,
        ),
    )

    print("Took", time.time() - st)

    print("Loading validation data")

    dataset_test = torchvision.datasets.MNIST(
        root=args.data_path,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]),
    )

    print("Creating data loaders")
    loader_g = torch.Generator()
    loader_g.manual_seed(args.seed)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def load_DVS_Gesture( args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data_path, 'train'),
        transform=presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            mean=(0.5,),
            std=(0.5,),
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        ),
    )

    print("Took", time.time() - st)

    print("Loading validation data")

    dataset_test = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data_path, 'test'),
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]),
    )

    print("Creating data loaders")
    loader_g = torch.Generator()
    loader_g.manual_seed(args.seed)

    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def get_cache_path( filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_ImageNet(args):
    # Data loading code
    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if not args.prototype:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
            )
        else:
            if args.weights:
                weights = prototype.models.get_weight(args.weights)
                preprocessing = weights.transforms()
            else:
                preprocessing = prototype.transforms.ImageNetEval(
                    crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
                )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    loader_g = torch.Generator()
    loader_g.manual_seed(args.seed)

    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler