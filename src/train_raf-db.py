import warnings
warnings.filterwarnings("ignore")
from apex import amp
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os, torch
import argparse
import Networks
from dataset import RafDataSet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None, help='Pretrained weights')
    # parser.add_argument('--pretrained', type=str, default='models/resnet18-5c106cde.pth', help='Pretrained weights')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=70, help='Total training epochs.')
    return parser.parse_args()

        
def run_training():
    args = parse_args()

    model = Networks.ResNet18_ARM___RAF()
    # print(model)
    print("batch_size:", args.batch_size)
            
    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained
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
    
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

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
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model = model.cuda()
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    CE_criterion = torch.nn.CrossEntropyLoss()


    best_acc = 0
    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):

            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            outputs, alpha = model(imgs)
            targets = targets.cuda()

            CE_loss = CE_criterion(outputs, targets)
            loss = CE_loss
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            
            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
                

        acc = correct_sum.float() / float(train_dataset.__len__())
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
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))

            if acc > 0.90 and acc > best_acc:
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('models/RAF-DB', "epoch" + str(i) + "_acc" + str(acc) + ".pth"))
                print('Model saved.')
            if acc > best_acc:
                best_acc = acc
                print("best_acc:" + str(best_acc))

            
if __name__ == "__main__":                    
    run_training()
