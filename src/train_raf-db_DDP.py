import warnings
warnings.filterwarnings("ignore")
from apex import amp
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import os, torch
import image_utils
import argparse, random
import Networks
import residual
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None, help='Pretrained weights')
    # parser.add_argument('--pretrained', type=str, default='models/resnet18-5c106cde.pth', help='Pretrained weights')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=70, help='Total training epochs.')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist-backend', default='gloo', type=str)
    return parser.parse_args()
    
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1   # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        
        self.file_paths = []
        # use raf-db aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]            # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx

        
def run_training():
    args = parse_args()
    ngpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = ngpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method="env://")
        synchronize()

    model = Networks.ResNet18_ARM___RAF()
    # model = residual.build_resnet('resnet50', 'classic', 7, verbose=False)
    # print(model)
    print("batch_size:", args.batch_size)

    model.cuda()

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained_state_dict = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        model_state_dict = model.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            model_state_dict[key] = pretrained_state_dict[key]
            total_keys+=1
            if key in model_state_dict:
                loaded_keys+=1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        model.load_state_dict(model_state_dict, strict = False)
        
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1))])
    
    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=(train_sampler is None),
                                               pin_memory=True,
                                               sampler=train_sampler)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    val_num = val_dataset.__len__()
    print('Validation set size:', val_num)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)
    
    params = model.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    print(optimizer)

    cudnn.benchmark = True
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    CE_criterion = torch.nn.CrossEntropyLoss()


    best_acc = 0
    for i in range(1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(i)
        train_loss = 0.0
        correct_sum = 0
        target_sum = 0
        iter_cnt = 0
        model.train()
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):

            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            outputs, alpha = model(imgs)
            targets = targets.cuda()
            target_sum += targets.size(0)

            CE_loss = CE_criterion(outputs, targets)
            loss = CE_loss
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
                

        acc = correct_sum.float() / target_sum
        train_loss = train_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f' %
              (i, acc, train_loss, optimizer.param_groups[0]["lr"]))
        scheduler.step()

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                outputs, _ = model(imgs.cuda())
                targets = targets.cuda()

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                
            running_loss = running_loss/iter_cnt   
            acc = bingo_cnt.float()/float(val_num)
            acc = np.around(acc.numpy(), 4)
            if get_rank() == 0:
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))

            if acc > 0.90 and acc > best_acc and get_rank() == 0:
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('models/RAF-DB', "epoch" + str(i) + "_acc" + str(acc) + ".pth"))
                print('Model saved.')
            if acc > best_acc and get_rank() == 0:
                best_acc = acc
                print("best_acc:" + str(best_acc))

            
if __name__ == "__main__":                    
    run_training()
