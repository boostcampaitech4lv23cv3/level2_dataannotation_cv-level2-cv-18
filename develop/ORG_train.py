import sys
import os
import errno
import os.path as osp
import time
import math
import json
from datetime import datetime, timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from glob import glob
import cv2

import numpy as np
import random

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from detect import detect
from deteval import calc_deteval_metrics

import wandb


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--wandb_name', type=str, default='Unnamed Test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_val', type=str2bool, default=True)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--load_from', type=str, default=None)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    if args.use_val == True and osp.isfile(osp.join(args.data_dir, 'ufo/val.json')) == False:
        print('Not found: val.json → Please set use_val=False or create val.json!')
        print('[Warning]: Force reset use_val=False')
        args.use_val = False

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, wandb_name, seed, use_val, val_interval, early_stop, load_from):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=np.random.seed(seed))

    if use_val:
        gt_bboxes_dict = dict()
        transcriptions_dict = dict()
        with open(osp.join(data_dir, 'ufo/val.json')) as json_file:
            file_contents = json_file.read()
            annotation_json = (json.loads(file_contents))['images']
            for image_name, ann_info in annotation_json.items():
                gt_bboxes_dict[image_name] = list()
                transcriptions_dict[image_name] = list()
                for word in  ann_info['words'].values():
                    gt_bboxes_dict[image_name].append(word['points'])
                    transcriptions_dict[image_name].append(word['transcription'])
        print("Load validation image...")
        val_images = []
        for image_name in tqdm(gt_bboxes_dict.keys()):
            if not osp.isfile(osp.join(data_dir, f'images/{image_name}')):
                del gt_bboxes_dict[image_name]
                del transcriptions_dict[image_name]
                continue
            val_images.append(cv2.imread(osp.join(data_dir, f'images/{image_name}'))[:, :, ::-1])
        val_num_batches = math.ceil(len(gt_bboxes_dict) / batch_size)

    model = EAST()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=0.01)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    if load_from and osp.isfile(load_from):
        try:
            checkpoint = torch.load(load_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            model.load_state_dict(torch.load(load_from))
        print(f"Loaded from: [{load_from}]")

    stop_cnt = 0
    best_score = 0
    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
                wandb.log({
                    'Train/Cls loss': extra_info['cls_loss'], 'Train/Angle loss': extra_info['angle_loss'],
                    'Train/IoU loss': extra_info['iou_loss']
                })

        scheduler.step()

        if stop_cnt == 0 :
            print('Mean loss: {:.4f} | Elapsed time: {}'.format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        else:
            print('Mean loss: {:.4f} | Elapsed time: {} | no more best count : {}'.format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start), stop_cnt))

        if use_val and (epoch + 1) % val_interval == 0:
            val_start = time.time()
            model.eval()
            with torch.no_grad():
                pred_bboxes = list()
                for step in tqdm(range(val_num_batches)):
                    images = val_images[batch_size * step:batch_size * (step + 1)]
                    pred_bboxes.extend(detect(model, images, input_size))

                pred_bboxes_dict = {image_fname:bboxes for image_fname, bboxes in zip(gt_bboxes_dict.keys(), pred_bboxes)}
                ret = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict, verbose=True)
                print(" ".join([f"F1: {ret['total']['hmean']:.4f}",
                                f"Precision: {ret['total']['precision']:.4f}",
                                f"Recall: {ret['total']['recall']:.4f}",
                                f"| Elapsed time: {timedelta(seconds=time.time() - val_start)}"
                              ]))

                wandb.log({
                    'Val/Precision': ret['total']['precision'], 'Val/Recall': ret['total']['recall'],
                    'Val/F1': ret['total']['hmean']
                })
            f1_score = ret['total']['hmean']

            if best_score < f1_score :
                best_score = f1_score
                print(f'New Best Model -> Epoch [{epoch+1}] / best_score : [{best_score}]')
                best_pth_name = f'{(wandb_name.replace(" ","_")).lower()}_best_model.pth'
                ckpt_fpath = osp.join(model_dir, best_pth_name)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            }, ckpt_fpath)
                symlink_force(best_pth_name, osp.join(model_dir, "best_model.pth"))
                stop_cnt = 0
            
            else:
                stop_cnt +=1

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            now = datetime.now()
            pth_name = f'{(wandb_name.replace(" ","_")).lower()}_{epoch+1}epoch_{now.strftime("%y%m%d_%H%M%S")}.pth'

            ckpt_fpath = osp.join(model_dir, pth_name)
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        }, ckpt_fpath)
            symlink_force(pth_name, osp.join(model_dir, "latest.pth"))


        if stop_cnt > early_stop :
            print(f'no more best model training | Training is over')
            break


def main(args):
    wandb.init(project="OCR Data annotation",
               entity="light-observer",
               name=args.wandb_name
              )
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
