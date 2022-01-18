import os
import torch
import torchvision.transforms as transforms

from Utils.folder import ImageFolder
import numpy as np
import cv2


def generate_dataloader(args, local_code_path):
    # Data loading code
    if args.dataset_name == "office31":
        traindir_s = os.path.join(local_code_path, "Data/office31/" + args.src + "/images")
        traindir_t = os.path.join(local_code_path, "Data/office31/" + args.tar + "/images")
        valdir_t = os.path.join(local_code_path, "Data/office31/" + args.tar_t + "/images")
    else:
        traindir_s = os.path.join(local_code_path, "Data/%s/%s" % (args.dataset_name, args.src))
        traindir_t = os.path.join(local_code_path, "Data/%s/%s" % (args.dataset_name, args.tar))
        valdir_t = os.path.join(local_code_path, "Data/%s/%s" % (args.dataset_name, args.tar_t))

    # transformation on the training data during training
    data_transform_train = transforms.Compose([
        # transforms.Resize((256, 256)), # spatial size of vgg-f input
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # transformation on the test data during test
    data_transform_test = transforms.Compose([
        transforms.Resize(256),  # spatial size of vgg-f input
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
       
    source_train_dataset = ImageFolder(root=traindir_s, transform=data_transform_train)
    source_test_dataset = ImageFolder(root=traindir_s, transform=data_transform_test)
    target_train_dataset = ImageFolder(root=traindir_t, transform=data_transform_train)
    target_test_dataset = ImageFolder(root=valdir_t, transform=data_transform_test)
    
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    source_test_loader = torch.utils.data.DataLoader(
        source_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return source_train_loader, target_train_loader, source_test_loader, target_test_loader


def _random_affine_augmentation(x):
    M = np.float32([[1 + np.random.normal(0.0, 0.1), np.random.normal(0.0, 0.1), 0],
                    [np.random.normal(0.0, 0.1), 1 + np.random.normal(0.0, 0.1), 0]])
    rows, cols = x.shape[1:3]
    dst = cv2.warpAffine(np.transpose(x.numpy(), [1, 2, 0]), M, (cols, rows))
    dst = np.transpose(dst, [2, 0, 1])
    return torch.from_numpy(dst)


def _gaussian_blur(x, sigma=0.1):
    ksize = int(sigma + 0.5) * 8 + 1
    dst = cv2.GaussianBlur(x.numpy(), (ksize, ksize), sigma)
    return torch.from_numpy(dst)
