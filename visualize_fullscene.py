import numpy as np
import torchvision
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision.ops
from retinanet.dataloader import CSVDataset, Resizer


print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a '
                                                 'RetinaNet network.')
    parser.add_argument('csv_classes', help='Path to file containing class list')
    parser.add_argument('csv_val', help='Path to file containing validation annotations')
    parser.add_argument('model', help='Path to model (.pt) file.')
    parser.add_argument('outputdir')
    parser.add_argument('--fullscene', type=bool, default=True)
    parser.add_argument('--detthresh', help='detection threshold for visualizing', type=float,
                        default=0.0)
    parser.add_argument('--score_thresh', help='score threshold to discard background/reduce '
                        'runs processing time', type=float, default=0.05)
    parser.add_argument('--iou_nms1', help='iou for nms used during validation and inference',
                        type=float, default=0.3)
    parser.add_argument('--iou_nms2', type=float, default=0.5)
    parser.add_argument('--pix_overlap', type=int, default=200,
                        help='number of pixel overlapping between patches')

    args = parser.parse_args()

    if not os.path.isdir(args.outputdir): os.makedirs(args.outputdir)

    if args.fullscene:
        dataset_val = CSVDataset(train_file=args.csv_val, class_list=args.csv_classes)
    else:
        dataset_val = CSVDataset(train_file=args.csv_val, class_list=args.csv_classes,
                                 transform=transforms.Compose([Resizer()]))

    dataloader_val = DataLoader(dataset_val, num_workers=1, shuffle=False)

    AllImgsAllDets = []

    for idx, data in enumerate(dataloader_val):
        iid = os.path.basename(dataset_val.image_names[idx])[:16]
        data['img'] = data['img'].permute(0, 3, 1, 2)
        _, ens, totalrows, totalcols = data['img'].shape
        nonoverlap = 800 - args.pix_overlap
        ufpatches = data['img'].unfold(2, 800, nonoverlap).unfold(3, 800, nonoverlap)
        _, _, numrow_p, numcol_p, _, _ = ufpatches.shape
        patches = ufpatches.contiguous().view(-1, 1, 800, 800)

        # load the weight
        retinanet = torch.load(args.model)

        retinanet = retinanet.cuda()
        retinanet.visualize = True
        retinanet.score_thresh = args.score_thresh
        retinanet.iou_nms1 = args.iou_nms1
        retinanet.eval()

        Allbbox = torch.tensor([])
        Allscore = torch.tensor([])
        Allclassification = torch.tensor([])
        Allclassscore = torch.tensor([])
        for patchidx in range(patches.shape[0]):
            patch = patches[patchidx:patchidx + 1, ...]
            with torch.no_grad():
                # st = time.time()
                class_scores, transformed_anchors = retinanet(patch.cuda().float())
                # print('Elapsed time: {}'.format(time.time()-st))
                if list(class_scores.shape) == [0]:
                    print('No detections')
                    continue
                scores, classification = class_scores.max(dim=1)
                # Compile all detections
                Allscore = torch.cat((Allscore, scores.cpu().float()))
                Allclassscore = torch.cat((Allclassscore, class_scores.cpu().float()))
                Allclassification = torch.cat((Allclassification, classification.cpu().
                                               float()))
                shifted_anchors = torch.empty(transformed_anchors.shape)
                shifted_anchors[:, [0, 2]] = transformed_anchors[:, [0, 2]].cpu() + \
                                             patchidx % numcol_p * nonoverlap
                shifted_anchors[:, [1, 3]] = transformed_anchors[:, [1, 3]].cpu() + \
                                             patchidx // numcol_p * nonoverlap
                Allbbox = torch.cat((Allbbox, shifted_anchors))


        # save out detections to numpy file
        if list(Allbbox.shape) == [0]:
            np.save(os.path.join(args.outputdir, '%s.npy' % iid), np.zeros((0, 7), dtype=np.float32))
        else:
            Allcenter = torch.cat((torch.mean(Allbbox[:, [0, 2]], dim=1, keepdim=True),
                                   torch.mean(Allbbox[:, [1, 3]], dim=1, keepdim=True)), dim=1)
        anchors_nms_idx = torchvision.ops.nms(Allbbox, Allscore, args.iou_nms2)
        
        # Alldetections is np array [detection scores, Allcenter, Allclassscore]
        Alldetections = np.hstack((Allscore[anchors_nms_idx, None].numpy(),
                                   Allbbox[anchors_nms_idx, :].numpy(),
                                   Allclassscore[anchors_nms_idx, :].numpy()))
        np.save(os.path.join(args.outputdir, '%s.npy' % iid), Alldetections)

        # reformat results for mAP score
        Allbbox = Allbbox[anchors_nms_idx]
        Allclassification = Allclassification[anchors_nms_idx]
        Allscore = Allscore[anchors_nms_idx]
        topbbox = Allbbox[Allscore >= args.detthresh]
        topclassification = Allclassification[Allscore >= args.detthresh]
        topscore = Allscore[Allscore >= args.detthresh]
        thisImgAllDets = [torch.cat((topbbox[topclassification == 0],
                                    topscore[topclassification == 0, None]), dim=1),
                          torch.cat((topbbox[topclassification == 1],
                                    topscore[topclassification == 1, None]), dim=1)]
        AllImgsAllDets.append(thisImgAllDets)

        # Visualize the whole scene
        img = np.array(255 * data['img'][0, 0, ...])
        img[img < 0] = 0
        img[img > 255] = 255
        fullscene = Image.fromarray(img).convert(mode='RGB')
        im_draw = ImageDraw.Draw(fullscene)

        for i in range(topscore.numpy().shape[0]):
            bbox1 = topbbox[i, :]
            im_draw.rectangle(list(bbox1), outline='red')
            x0y0 = list(bbox1[:2])
            x0y0[1] -= 10
            label_name1 = dataset_val.labels[int(topclassification[i])] + ', ' + str(topscore[i].cpu().numpy())[:4]
            im_draw.text(x0y0, label_name1, fill='yellow')

        fullscene.save(os.path.join(args.outputdir, '%s.png' % iid))



if __name__ == '__main__':
    main()
