from __future__ import print_function

import argparse
import os
from pickle import FALSE
import random
import time
import csv

import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

import random
from torchvision.utils import save_image

from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

import models
import preproc as pre
from losses import BeliefMatchingLoss
from metrics import compute_total_entropy, compute_max_prob, compute_differential_entropy, compute_mutual_information, \
    compute_precision
from utils import progress_bar, convert_to_rgb

parser = argparse.ArgumentParser(description='Meta model training')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_false', help='resume from checkpoint')
parser.add_argument('--base_model', default="ResNet18_AllFeatures", type=str, help='model type (default: LeNet)')
parser.add_argument('--base_epoch', default=200, type=int, help='total epochs to train base model')
parser.add_argument('--meta_model', default="Resnet18_meta_model", type=str,
                    help='model type (default: LeNet)')
parser.add_argument('--name', default='CIFAR10_OOD', type=str, help='name of run')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='name of run')
parser.add_argument('--seed_trail', default=1, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=30, type=int, help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lambda_KL', default=1e-3, type=float, help='lambda for KL term in ELBO loss')
#Added the following parser
parser.add_argument('--early_stop', default=FALSE, type=bool, help='Early Stop')
parser.add_argument('--forwardhook', default = FALSE, type=bool,  help='If your architecture do not return the feature layer then make it true')
parser.add_argument('--sigma', default=5e-4, type=float, help='noise level')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()

args.forwardhook = False

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_auroc_MI = 0
best_auroc_tot_ent = 0

if args.dataset == 'MNIST':
    args.fea_dim = [6 * 14 * 14, 5 * 5 * 16]
elif args.dataset == 'BACH':
    # args.fea_dim = [1048576,524288,262144,131072]
    args.fea_dim = [65536,131072,65536,131072]
else:
    args.fea_dim = [4096//1, 2048//1,1024//1,512]#,10]#, 512]

if use_cuda:
    torch.manual_seed(args.seed_trail)
    torch.cuda.manual_seed_all(args.seed_trail)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed_trail)
    random.seed(args.seed_trail)
    os.environ['PYTHONHASHSEED'] = str(args.seed_trail)
print('==> Resuming from checkpoint..')
assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.t7CIFAR10_0_80',#Bach__1e-3_1_1e-1_1_150',
                        map_location=torch.device('cpu') if not use_cuda else None)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        print(len(self.data_frame))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label



class CutMixCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root="./data", train=True, download=True, beta=1.0, transform=None):
        """
        Creates a new dataset from CIFAR-10 where each image is a CutMix of two different classes.

        Args:
            root (str): CIFAR-10 root directory.
            train (bool): Use train or test split.
            download (bool): Whether to download CIFAR-10.
            beta (float): Beta distribution parameter for patch area ratio.
            transform (callable): Transform applied after mixing.
        """
        self.base_dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.beta = beta
        self.transform = transform
        self.cutmix_prob = 0.5
        self.num_classes = 11
        self.labels_onehot = torch.eye(self.num_classes)[self.base_dataset.targets]

    def __len__(self):
        return len(self.base_dataset)

    def rand_bbox(self, width, height, lam):
        """Generate random bbox coordinates for CutMix."""
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(width * cut_rat)
        cut_h = int(height * cut_rat)

        # uniform center
        cx = np.random.randint(width)
        cy = np.random.randint(height)

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)

        return bbx1, bby1, bbx2, bby2

    def __getitem__(self, idx):
#         img1, label1 = self.base_dataset[idx]
#         img2_idx = random.choice([i for i in range(len(self.base_dataset)) if self.base_dataset.targets[i] != label1])
#         img2, label2 = self.base_dataset[img2_idx]
        
#         # Convert to NumPy (H, W, C)
#         img1 = np.array(img1, dtype=np.uint8)
#         img2 = np.array(img2, dtype=np.uint8)

# #         to_tensor = transforms.ToTensor()
# #         img1 = to_tensor(img1)
# #         img2 = to_tensor(img2)

#         H, W, C = img1.shape

#         # Sample lambda from Beta distribution
#         lam = np.random.beta(self.beta, self.beta)
#         bbx1, bby1, bbx2, bby2 = self.rand_bbox(W, H, lam)

#         # Replace patch in-place
#         img1[bby1:bby2, bbx1:bbx2, :] = img2[bby1:bby2, bbx1:bbx2, :]

#         # Adjust lambda to actual area ratio
#         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

#         # Mix labels
#         y1 = self.labels_onehot[label1]
#         y2 = self.labels_onehot[label2]
#         #mixed_label = lam * y1 + (1 - lam) * y2
#         mixed_label = (lam * y1 + (1 - lam) * y2).round().long()
        
#         img1 = Image.fromarray(img1)
        
#         if self.transform:
#             img1 = self.transform(img1)

        img1, label1 = self.base_dataset[idx]

        img1 = np.array(img1, dtype=np.uint8)
        H, W, C = img1.shape

        # Randomly decide to apply CutMix
        if random.random() < self.cutmix_prob:
            # Select another image from a different class
            img2_idx = random.choice(
                [i for i in range(len(self.base_dataset)) if self.base_dataset.targets[i] != label1]
            )
            img2, label2 = self.base_dataset[img2_idx]
            img2 = np.array(img2, dtype=np.uint8)

            # Sample lambda from Beta distribution
            lam = np.random.beta(self.beta, self.beta)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(W, H, lam)

            # Apply CutMix patch
            img1[bby1:bby2, bbx1:bbx2, :] = img2[bby1:bby2, bbx1:bbx2, :]

            # Assign new label 10 for CutMix
            label = torch.tensor(10)
        else:
            # Keep original image and label
            label = label1

        img1 = Image.fromarray(img1)

        if self.transform:
            img1 = self.transform(img1)

        
        
        
        

        return img1, label


'''
Processing data
'''
sigma = args.sigma
print('==> Preparing data..')
# Noisy validation set for OOD
if args.dataset == 'MNIST':
    transform_noise = transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        pre.GaussianFilter(),
    ])
elif args.dataset == 'BACH':
    transform_noise = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        pre.GaussianFilter(),
        pre.ContrastRescaling(),
    ])
else:
    transform_noise = transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        #pre.PermutationNoise(),
        transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * sigma, 0, 1)),
        #pre.GaussianFilter(),
        #pre.ContrastRescaling(),
    ])

if args.dataset == 'CIFAR10':
    print('CIFAR10')
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * sigma, 0, 1)),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    #cutmix_train = CutMixCIFAR10(train=True, beta=1.0, transform=transform_train)
    dataset = datasets.CIFAR10(root='~/data/CIFAR10', train=True, download=True, transform=transform_train)
    #dataset = datasets.SVHN(root='~/data/SVHN', split='train', download=True, transform=transform_train)
    dataset_val = datasets.CIFAR10(root='~/data/CIFAR10', train=True, download=True, transform=transform_train)
    #dataset_val = datasets.SVHN(root='~/data/SVHN', split='train', download=True, transform=transform_noise)
    dataset_noise = datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform_test)
    dataset_ood = datasets.MNIST(root='./data',train=True,download=True,transform=transform_test)
    #dataset_ood = datasets.SVHN(root='~/data/SVHN', split='test', download=True, transform=transform_test)
    #dataset_ood = datasets.CIFAR10(root='~/data/CIFAR10', train=False, download=True, transform=transform_test)
    ood_loader = torch.utils.data.DataLoader(dataset_ood, batch_size=20, shuffle=False, num_workers=8)
    num_total_data = int(len(dataset))
    random.seed(args.seed_trail)


    # ID_CLASSES  = [0, 1, 2, 3, 4,5,6]
    # OOD_CLASSES = [7, 8, 9]
    # id_indices = []
    # ood_indices = []
    
    # for idx, label in enumerate(dataset.targets):
    
    #     if label in ID_CLASSES:
    #         id_indices.append(idx)
    
    #     elif label in OOD_CLASSES:
    #         ood_indices.append(idx)

    data_list = list(range(num_total_data))
    random.shuffle(data_list)

    # random.seed(args.seed_trail)

    # random.shuffle(id_indices)
    
    # split = int(0.8 * len(id_indices))
    
    # train_list = id_indices[:split]
    # val_list   = id_indices[split:]
    
    train_list = data_list[:40000]
    val_list = data_list[40000:]
    trainset = data.Subset(dataset, train_list)
    #cutmixset = data.Subset(cutmix_train,train_list)
    valset = data.Subset(dataset_val, val_list)
    valset_noise = data.Subset(dataset_noise, val_list)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valloader_noise = torch.utils.data.DataLoader(valset_noise, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=4)
    #cutmixloader = torch.utils.data.DataLoader(cutmixset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # oodset = data.Subset(dataset, [
    # idx for idx, label in enumerate(dataset.targets)
    # if label in OOD_CLASSES
    # ])

    # ood_loader = torch.utils.data.DataLoader(
    # oodset,
    # batch_size=20,
    # shuffle=False,
    # num_workers=8
    # )
    testset = datasets.CIFAR10(root='~/data/CIFAR10', train=False, download=True, transform=transform_test)
    #testset = datasets.SVHN(root='~/data/SVHN', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=8)

elif args.dataset == 'CIFAR100':
    print('CIFAR100')
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = datasets.CIFAR100(root='~/data/CIFAR100', train=True, download=True, transform=transform_train)
    dataset_val = datasets.CIFAR100(root='~/data/CIFAR100', train=True, download=False, transform=transform_test)
    dataset_noise = datasets.CIFAR100(root='~/data/CIFAR100', train=True, download=False, transform=transform_noise)
    
    num_total_data = int(len(dataset))
    random.seed(args.seed_trail)
    data_list = list(range(num_total_data))
    random.shuffle(data_list)
    train_list = data_list[:40000]
    val_list = data_list[40000:]
    trainset = data.Subset(dataset, train_list)
    valset = data.Subset(dataset_val, val_list)
    valset_noise = data.Subset(dataset_noise, val_list)
    
    dataset_ood = datasets.MNIST(root='./data',train=True,download=True,transform=transform_test)
    ood_loader = torch.utils.data.DataLoader(dataset_ood, batch_size=20, shuffle=False, num_workers=8)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    valloader_noise = torch.utils.data.DataLoader(valset_noise, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=8)
    testset = datasets.CIFAR100(root='~/data/CIFAR100', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

elif args.dataset == 'MNIST':
    if args.augment:
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize((1 / 2, 1 / 2, 1 / 2), (1 / 2, 1 / 2, 1 / 2))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize((1 / 2, 1 / 2, 1 / 2), (1 / 2, 1 / 2, 1 / 2))
        ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize((1 / 2, 1 / 2, 1 / 2), (1 / 2, 1 / 2, 1 / 2))
    ])
    dataset = datasets.MNIST(root='~/data/MNIST', train=True, download=True, transform=transform_train)
    dataset_val = datasets.MNIST(root='~/data/MNIST', train=True, download=True, transform=transform_test)
    dataset_noise = datasets.MNIST(root='~/data/MNIST', train=True, download=False, transform=transform_noise)
    num_total_data = int(len(dataset))
    random.seed(args.seed_trail)
    data_list = list(range(num_total_data))
    random.shuffle(data_list)
    train_list = data_list[:50000]
    val_list = data_list[50000:]
    trainset = data.Subset(dataset, train_list)
    valset = data.Subset(dataset, val_list)
    valset_noise = data.Subset(dataset_noise, val_list)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    valloader_noise = torch.utils.data.DataLoader(valset_noise, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=8)
    testset = datasets.MNIST(root='~/data/MNIST', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

elif args.dataset == 'BACH':
    csv_file_path = "./BACH_Dataset/data_labels.csv"
    image_folder_path = "./BACH_Dataset/train_image"

    image_transforms = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=3),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        
    ])

    dataset = ImageDataset(csv_file=csv_file_path, root_dir=image_folder_path, transform=image_transforms)
    dataset_noise = ImageDataset(csv_file=csv_file_path, root_dir=image_folder_path, transform=transform_noise)
    num_total_data = int(len(dataset))
    random.seed(args.seed_trail)

    data_list = list(range(num_total_data))
    random.shuffle(data_list)
    random.shuffle(data_list)
    
    train_list = data_list[:200]
    val_list = data_list[200:300]
    test_list = data_list[300:]
    trainset = data.Subset(dataset, train_list)
    valset = data.Subset(dataset, val_list)
    valset_noise = data.Subset(dataset_noise, val_list)
    testset = data.Subset(dataset, test_list)

    # def extract_labels(dataset):
    #     labels = []
    #     for idx in range(len(dataset)):
    #         label = dataset[idx][1]  # Assuming the label is at index 1, modify if needed
    #         labels.append(label)
    #     return labels

    # # Extract labels from the datasets
    # labels = extract_labels(testset)

    # # Create a histogram of the labels
    # histogram = np.histogram(labels, bins=np.arange(0, 5))  # Adjust num_classes as needed

    # # Plot the histogram
    # plt.bar(histogram[1][:-1], histogram[0], width=0.5)
    # plt.xlabel('Label')
    # plt.ylabel('Frequency')
    # plt.title('Label Histogram')
    # plot_path = "./"
    # plt.savefig(os.path.join(plot_path, 'testhisto.png'))

    # exit()

    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valloader_noise = torch.utils.data.DataLoader(valset_noise, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=4)




class ModifiedResNet18(nn.Module):
    def __init__(self, original_model, dropout_prob=0.5):
        super(ModifiedResNet18, self).__init__()
        
        # Extract the layers from the original model
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
        # Add dropout to convolutional layers
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                module.add_module("dropout", nn.Dropout2d(p=dropout_prob))
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 4)  # Change 1000 to the number of output classes
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

'''
Preparing model
'''
if args.dataset == 'CIFAR100':
    base_net = models.__dict__[args.base_model](num_classes=100) #(16, 4, 100, 3)
elif args.dataset == 'BACH':
    base_net = torchvision.models.resnet18(pretrained=True)
    dropout_prob = 0.4  # Adjust the dropout probability as needed
    base_net = ModifiedResNet18(base_net, dropout_prob)



else:
    base_net = models.__dict__[args.base_model]()

if args.dataset in ['CIFAR10', 'CIFAR100']:
    meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
                                                fea_dim3=args.fea_dim[2], fea_dim4=args.fea_dim[3]) #, fea_dim5=args.fea_dim[4])
elif args.dataset == 'BACH':
    ## Import model here
    # meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
                                                # fea_dim3=args.fea_dim[2], fea_dim4=args.fea_dim[3])
    meta_net = models.__dict__[args.meta_model](fea_dim2=args.fea_dim[1],
                                                fea_dim3=args.fea_dim[2], fea_dim4=args.fea_dim[3])


else:
    meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1])
if use_cuda:
    print('Using CUDA..')
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    base_net = base_net.cuda()
    meta_net = meta_net.cuda()

base_net.load_state_dict(checkpoint['net'])
base_net.eval()
for k, v in base_net.named_parameters():
    v.requires_grad = False

param_group = []
meta_net.eval()

optimizer = optim.SGD(meta_net.parameters(), momentum=0.9, weight_decay=args.decay, lr=args.lr)

if not os.path.isdir('results_meta_new'):
    os.mkdir('results_meta_new')
# logname = ('results_meta/log_' + meta_net.__class__.__name__ + '_' + args.name + '_' + str(args.seed_trail) + '.csv')
logname = ('results_meta_new/log_' + args.name + '_resnet18_0246_' + str(args.seed_trail) + '.csv')

'''
Training Meta-model
'''
vi_loss = BeliefMatchingLoss(args.lambda_KL, 1)

def getting_features(input_tensor):
  input_tensor = input_tensor.cuda()
  # print(input_tensor.shape)
  batch_size = input_tensor.size(0)
  # a dict to store the activations
  activation = {}
  def getActivation(name):
    # the hook signature
    def hook(model, input, output):
      activation[name] = output.detach()
    return hook

  # register forward hooks on the layers of choice
#   h1 = base_net.features[4][1].bn2.register_forward_hook(getActivation('layer1'))
#   h2 = base_net.features[5][1].bn2.register_forward_hook(getActivation('layer2'))
#   h3 = base_net.features[6][1].bn2.register_forward_hook(getActivation('layer3'))
#   h4 = base_net.features[7][1].bn2.register_forward_hook(getActivation('layer4'))

  # print(h1)

  # forward pass -- getting the outputs
  output, features = base_net(input_tensor)
  # print(input_tensor.shape)
  activation['layer2'] = features[1]
  activation['layer3'] = features[2]
  activation['layer4'] = features[3]

#   h1.remove()
#   h2.remove()
#   h3.remove()
#   h4.remove()
  # print("Activation Shape")
#   print(activation['layer1'].shape)
#   print(activation['layer2'].shape)
#   print(activation['layer3'].shape)
#   print(activation['layer4'].shape)

#   avg_pooling_l1 = nn.AvgPool2d(kernel_size=4, stride=4)
#   activation['layer1'] = avg_pooling_l1(activation['layer1'])
  avg_pooling_l2 = nn.AvgPool2d(kernel_size=2, stride=2)
  activation['layer2'] = avg_pooling_l2(activation['layer2'])
  avg_pooling_l3 = nn.AvgPool2d(kernel_size=2, stride=2)
  activation['layer3'] = avg_pooling_l2(activation['layer3'])





#   g1 = torch.flatten(activation['layer1']).view(batch_size, -1)
  g2 = torch.flatten(activation['layer2']).view(batch_size, -1)
  g3 = torch.flatten(activation['layer3']).view(batch_size, -1)
  g4 = torch.flatten(activation['layer4']).view(batch_size, -1)
  # print("G! SHAPE")
#   print(g1.shape)
#   print(g2.shape)
#   print(g3.shape)
#   print(g4.shape)
  # combined_tensor = torch.cat((g1, g2, g3, g4), dim=1)
  # print(combined_tensor.shape)
  return [g2,g3,g4]


def compute_logits_and_loss(xs, ys, compute_loss=False):
    loss = torch.Tensor([0])
    if args.forwardhook:
        fea_list = getting_features(xs)
    else:
        if args.forwardhook:
            fea_list = getting_features(xs)
        else:    
            output, fea_list = base_net(xs)
            probs = F.softmax(output, dim=1)
            # max probability
            max_prob, _ = torch.max(probs, dim=1, keepdim=True)

            # entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
            features = output #torch.cat([output, max_prob, entropy], dim=1)

    #print(*fea_list[4].shape) # (0123) 128 64 8 8 (4567) 128  128 4 4
    idx = random.choice([1,2,3,4,5,6])
    if compute_loss == False:
        idx = 7
    logits, var_score,f1,f2 = meta_net(fea_list[4][:, :, :, :],fea_list[6][:, :, :, :],fea_list[8][:, :, :, :],fea_list[10][:, :, :, :], idx)#,features)
    #print(logits.shape,ys.shape)
    if compute_loss:
        loss = vi_loss(logits, ys,f1,f2)
    else:
        logits = logits[:, :10]
    # print(logits)
    return logits, loss , var_score


def train(epoch):
    print('\nEpoch: %d' % epoch)
    base_net.eval()
    meta_net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (xs, ys) in enumerate(trainloader):
        total += ys.size(0)
        if use_cuda:
            xs, ys = xs.cuda(), ys.cuda()
        #print(args.forwardhook)
        logits, loss,_ = compute_logits_and_loss(xs, ys, compute_loss=True)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        _, predicted = torch.max(logits.data, 1)
        correct += predicted.eq(ys).cpu().sum()

        progress_bar(current=batch_idx,
                     total=len(trainloader),
                     msg='Loss: %.3f |  Acc: %.3f%% (%d/%d)' % (
                         train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # for batch_idx, (xs, ys) in enumerate(cutmixloader):
    #     #total += ys.size(0)
    #     if use_cuda:
    #         xs, ys = xs.cuda(), ys.cuda()
    #     #print(args.forwardhook)
    #     logits, loss,_ = compute_logits_and_loss(xs, ys, compute_loss=True)

    #     train_loss += loss.item()

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
       
        # _, predicted = torch.max(logits.data, 1)
        # correct += predicted.eq(ys).cpu().sum()

    train_loss_final = train_loss / ((batch_idx+1))
    acc = 100. * correct / total

    return (train_loss_final, acc)


'''
Testing Meta-model
'''


def test(epoch):
    global best_acc
    base_net.eval()
    meta_net.eval()
    test_loss = 0
    correct = 0

    total_entropy = 0
    max_prob = 0
    mutual_info = 0
    diff_entropy = 0
    precision = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (xs, ys) in enumerate(testloader):
            total += ys.size(0)
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            logits, loss,_ = compute_logits_and_loss(xs, ys, compute_loss=True)
            test_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct += predicted.eq(ys.data).cpu().sum()
            # print(logits)
            # Uncertainty Criterion
            total_entropy += compute_total_entropy(logits).sum()
            max_prob += compute_max_prob(logits).sum()
            mutual_info += compute_mutual_information(logits).sum()
            diff_entropy += compute_differential_entropy(logits).sum()
            precision += compute_precision(logits).sum()

            # progress_bar(batch_idx, len(testloader),
            #              'Loss: %.3f | Acc: %.3f%% (%d/%d) | '
            #              'DEnt: %.3f | MI: %.3f | TotEnt: %.3f | MaxP: %.3f | Prec: %.3f' %
            #              (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,
            #               diff_entropy / total, mutual_info / total,
            #               total_entropy / total, max_prob / total, precision / total))
        # print(total)
        test_loss_final = test_loss / (batch_idx + 1)   
        acc = 100. * correct / total

        return (test_loss_final, acc)


'''
Validation for OOD task
'''
def UQ_validation():
    global best_auroc_MI
    global best_auroc_tot_ent
    base_net.eval()
    meta_net.eval()
    flag = True
    total = 0
    saved=False
    with torch.no_grad():
        # In distribution data
        for batch_idx, (xs, ys) in enumerate(valloader):
            if not saved:
                rand_idx = random.randint(0, xs.size(0) - 1)
                save_image(xs[rand_idx], "noisy_sample_idcifar.png")
#                 saved = True
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            logits, _, var_score = compute_logits_and_loss(xs, ys, compute_loss=False)

            # Uncertainty Criterion
            mutual_info = var_score #compute_mutual_information(logits)+ 0.1*var_score
            #print(mutual_info.shape)
            tot_ent = compute_total_entropy(logits)
            _, meta_predicted = torch.max(logits.data, 1)
            meta_correct = meta_predicted.ne(ys.data)
            if flag:
                all_label = torch.zeros((ys.size()[0]))
                all_mutual_info = mutual_info.data.cpu()
                all_tot_ent = tot_ent.data.cpu()
                all_meta_predicted = meta_correct.data.cpu()
                flag = False
            else:
                all_label = torch.cat((all_label, torch.zeros((ys.size()[0]))), 0)
                all_mutual_info = torch.cat((all_mutual_info, mutual_info.data.cpu()), 0)
                all_tot_ent = torch.cat((all_tot_ent, tot_ent.data.cpu()), 0)
                all_meta_predicted = torch.cat((all_meta_predicted, meta_correct.data.cpu()), 0)
        #print(all_tot_ent.shape)
        #print('ID mean entropy :', torch.mean(all_tot_ent,0)/np.log(10))
        torch.save(all_tot_ent,"id_tot_ents.pt")
        ood_entropy = torch.empty(0)
        # Out of distribution data
        for batch_idx, (xs, ys) in enumerate(ood_loader):
            if not saved:
                rand_idx = random.randint(0, xs.size(0) - 1)
                save_image(xs[rand_idx], "noisy_sample_oodmnist.png")
                saved = True
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()
            logits, _ ,var_score= compute_logits_and_loss(xs, ys, compute_loss=False)

            # Uncertainty Criterion
            mutual_info = var_score #compute_mutual_information(logits)+ 0.1*var_score
            tot_ent = compute_total_entropy(logits)
            ood_entropy = torch.cat((ood_entropy, tot_ent.data.cpu()), dim=0)
            all_label = torch.cat((all_label, torch.ones((ys.size()[0]))), 0)
            all_mutual_info = torch.cat((all_mutual_info, mutual_info.data.cpu()), 0)
            all_tot_ent = torch.cat((all_tot_ent, tot_ent.data.cpu()), 0)
    
    #print('OOD mean entropy :', torch.mean(ood_entropy,0)/np.log(10))
    torch.save(ood_entropy,"ood_tot_ents.pt")
    # ood Auroc score evaluated using mutual information
    auroc_MI = metrics.roc_auc_score(all_label.numpy(), all_mutual_info.numpy())
    auroc_tot_ent = metrics.roc_auc_score(all_label.numpy(), all_tot_ent.numpy())
    if auroc_MI > best_auroc_MI:
        checkpoint(auroc_MI, epoch)
        best_auroc_MI = auroc_MI
    if auroc_tot_ent > best_auroc_tot_ent:
        best_auroc_tot_ent = auroc_tot_ent

    return


'''
Inference the meta-model on OOD dataset (noisy images)
'''


def OOD(epoch):
    base_net.eval()
    meta_net.eval()

    total_entropy = 0
    max_prob = 0
    mutual_info = 0
    diff_entropy = 0
    precision = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (xs, ys) in enumerate(ood_loader):
            total += ys.size(0)
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            # test meta model
            logits, _ ,var_score = compute_logits_and_loss(xs, ys, compute_loss=False)

            # Uncertainty Criterion
            total_entropy += compute_total_entropy(logits).sum()
            max_prob += compute_max_prob(logits).sum()
            mutual_info += compute_mutual_information(logits).sum()
            diff_entropy += compute_differential_entropy(logits).sum()
            precision += compute_precision(logits).sum()
#             progress_bar(batch_idx, len(valloader_noise),
#                          'DEnt: %.3f | MI: %.3f | TotEnt: %.3f | MaxP: %.3f | Prec: %.3f'
#                          % (diff_entropy / total, mutual_info / total,
#                             total_entropy / total, max_prob / total, precision / total))
    return total_entropy/total, max_prob/total, mutual_info/total, diff_entropy/total, precision/total


def checkpoint(auroc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'meta_net': meta_net.state_dict(),
        'auroc': auroc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint_new'):
        os.mkdir('checkpoint_new')
    torch.save(state, './checkpoint_new/ckpt_resnet18_grbg.t7' + args.meta_model  + '_' + args.name + '_' + str(args.seed_trail))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    lr /= 10
    if epoch >= 20:
        lr /= 100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


'''
Main training process
'''
if __name__ == '__main__':
    
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss',  'train acc', 'test loss', 'test acc','TotEnt','MaxP','MI','DEnt','Precision'])
        time_start = time.perf_counter()
    # print(len(testloader))
    for epoch in range(start_epoch, args.epoch + 1):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        if args.name in ['SVHN_OOD','CIFAR10_OOD', 'CIFAR100_OOD', 'MNIST_OOD','BACH_OOD','BACH','cifar10_compressed']:
            total_entropy, max_prob, mutual_info, diff_entropy, precision = OOD(epoch)
            UQ_validation()
            print(best_auroc_MI)
            print(best_auroc_tot_ent)
            adjust_learning_rate(optimizer, epoch)
        elif args.dataset == "BACH":
            OOD(epoch)
            UQ_validation()
            adjust_learning_rate(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc,total_entropy, max_prob, mutual_info, diff_entropy, precision])
        
        



    print('Finished')
    training_time = time.perf_counter() - time_start
    print('Total training time', training_time)
    print("Best AUROC MI ", best_auroc_MI)
    print("Best AUROC tot_ent ", best_auroc_tot_ent)


    auroc_csv = "./auroc_resnet18_cifar10_compressed_04812.csv"


# Open the CSV file in append mode
    new_row = [ args.name + '_' + str(args.seed_trail), 'MI:', best_auroc_MI,'Tot ent:', best_auroc_tot_ent]

    with open(auroc_csv, mode='a', newline='') as file:
    # Create a CSV writer object
        writer = csv.writer(file)
    
        # Write the new row to the CSV file
        writer.writerow(new_row)
    
    
    if not args.early_stop:
            checkpoint(0, args.epoch)
    if args.name in ['CIFAR10_miss', 'CIFAR100_miss', 'MNIST_miss']:
        checkpoint(0, args.epoch)
