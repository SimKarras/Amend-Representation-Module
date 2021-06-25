import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
import argparse
import Networks
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from dataset import RafDataSet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('-p', '--plot_cm', action="store_true", help="Ploting confusion matrix.")
    return parser.parse_args()


def test():
    args = parse_args()
    model = Networks.ResNet18_ARM___RAF()

    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    data_transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_test)
    test_size = test_dataset.__len__()
    print('Test set size:', test_size)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    model = model.cuda()

    pre_labels = []
    gt_labels = []
    with torch.no_grad():
        bingo_cnt = 0
        model.eval()
        for batch_i, (imgs, targets, _) in enumerate(test_loader):
            outputs, _ = model(imgs.cuda())
            targets = targets.cuda()
            _, predicts = torch.max(outputs, 1)
            correct_or_not = torch.eq(predicts, targets)
            pre_labels += predicts.cpu().tolist()
            gt_labels += targets.cpu().tolist()
            bingo_cnt += correct_or_not.sum().cpu()

        acc = bingo_cnt.float() / float(test_size)
        acc = np.around(acc.numpy(), 4)
        print(f"Test accuracy: {acc:.4f}.")

    if args.plot_cm:
        cm = confusion_matrix(gt_labels, pre_labels)
        cm = np.array(cm)
        labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]  # 横纵坐标标签
        plot_confusion_matrix(cm, labels_name, 'RAF-DB', acc)


if __name__ == "__main__":                    
    test()
