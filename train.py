import os
import argparse
import collections
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from retinanet import model
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer

import csv_eval
from lr_finder import LRFinder

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

    parser = argparse.ArgumentParser(description='Training a RetinaNet network.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations')
    parser.add_argument('--csv_classes', help='Path to file containing class list')
    parser.add_argument('--csv_val', help='Path to file containing validation \
                        annotations')
    parser.add_argument("--depth", help='Resnet depth, must be one of \
                        18, 34, 50,101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs to run' ,type=int, default=100)
    parser.add_argument('--batch_size', help='Number of training sample per batch',
                        type=int, default=16)
    parser.add_argument('--score_thresh', help='score threshold to discard \
                        background/reduce nms processing time', default=0.05)
    parser.add_argument("--iou_nms1", help="iou for nms used during validation and \
                        inference", type=float, default=0.3)
    parser.add_argument('--lr', help='learning rate', type=float, default=6e-4)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--logfile')

    args = parser.parse_args(args)

    outputdir = os.path.dirname(args.logfile)
    if not os.path.isdir(outputdir): os.makedirs(outputdir)

    # Create the data loaders
    if args.csv_train is None:
        raise ValueError('Must provide --csv_train when training on CSV,')

    if args.csv_classes is None:
        raise ValueError('Must provide --csv_classes when training on CSV,')

    dataset_train = CSVDataset(train_file=args.csv_train, class_list=args.csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    if args.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=args.csv_val, class_list=args.csv_classes,
                                 transform=transforms.Compose([Normalizer(),
                                                               Resizer()]))

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  num_workers=3, collate_fn=collater, shuffle=True)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater,
                                    batch_sampler=sampler_val)

    # Create the model
    if args.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(),
                                   pretrained=args.pretrained)
    elif args.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(),
                                   pretrained=args.pretrained)
    elif args.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(),
                                   pretrained=args.pretrained)
    elif args.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(),
                                    pretrained=args.pretrained)
    elif args.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(),
                                    pretrained=args.pretrained)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.training = True
    retinanet.score_thresh = args.score_thresh
    retinanet.iou_nms1 = args.iou_nms1

    optimizer = optim.Adam(retinanet.parameters(), lr=args.lr)

    # # LR Finder
    # lr_finder = LRFinder(retinanet, optimizer, losses.FocalLossQ, device="cuda")
    # lr_finder.range_test(dataloader_train, end_lr=10, num_iter=1260, diverge_th=10)
    # Ir_finder.plot(skip_start=0, skip_end=3, show_lr=3e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    print("Num training images: {}".format(len(dataset_train)))

    for epoch_num in range(args.epochs):

        retinanet.train()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()

            classification_loss, regression_loss = retinanet([data['img'].cuda().float(),
                                                              data['annot']])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | '
                  'Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num,
                   iter_num, float(classification_loss), float(regression_loss),
                   np.mean(loss_hist)))

            del classification_loss
            del regression_loss

        if args.csv_val is not None:
            mAP = csv_eval.evaluate(dataset_val, retinanet)
            with open(args.logfile, mode='a') as f:
                f.write("mAP:\n")
                aps = []
                for i, label_name in enumerate(dataset_val.classes):
                    f.write('{}: {}| Count: {}\n'.format(label_name, mAP[i][0],mAP[i][1]))
                    aps.append(mAP[i][0])
                f.write('mAP: {}\n'.format(np.mean(aps)))

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module,
                   '{}/retinanet_{}.pt'.format(outputdir, epoch_num))
        torch.save(retinanet.module.state_dict(),
                   '{}/statedict_{}.pt'.format(outputdir, epoch_num))

    retinanet.eval()


if __name__ == '__main__':
    main()
