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

import numpy as np
import random

from east_dataset import EASTDataset
from dataset import SceneTextDataset, ValidSceneTextDataset
from model import EAST

from detect import get_bboxes
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--wandb_name', type=str, default='Unnamed Test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_val', type=str2bool, default=True)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=5)
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
    os.environ['PYTHONHASHSEED'] = str(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker)

    if use_val:
        val_dataset = ValidSceneTextDataset(data_dir, split='val', image_size=image_size, crop_size=image_size, color_jitter=False)
        val_dataset.load_image()
        print(f"Load valid data {len(val_dataset)}")
        valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=ValidSceneTextDataset.collate_fn)
        val_num_batches = math.ceil(len(val_dataset) / batch_size)

    model = EAST()
    if load_from and osp.isfile(load_from):
        try:
            checkpoint = torch.load(load_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            model.load_state_dict(torch.load(load_from))
        print(f"Loaded from: [{load_from}]")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=0.01)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    stop_cnt = 0
    best_score = 0
    for epoch in range(max_epoch):
        
        # Train
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        epoch_cls_loss, epoch_angle_loss, epoch_iou_loss = 0, 0, 0
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_cls_loss += extra_info['cls_loss']
                epoch_angle_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        wandb.log({
            'Train/Cls loss': epoch_cls_loss / num_batches,
            'Train/Angle loss': epoch_angle_loss / num_batches,
            'Train/IoU loss': epoch_iou_loss / num_batches,
            'Train/Loss': epoch_loss / num_batches,
        })

        if stop_cnt == 0 :
            print('Mean loss: {:.4f} | Elapsed time: {}'.format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        else:
            print('Mean loss: {:.4f} | Elapsed time: {} | no more best count : {}'.format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start), stop_cnt))

        # Validation
        if use_val and (epoch + 1) % val_interval == 0:
            val_epoch_loss, val_epoch_cls_loss, val_epoch_angle_loss, val_epoch_iou_loss, val_start = 0, 0, 0, 0, time.time()

            pred_bboxes_dict = dict()
            gt_bboxes_dict = dict()
            transcriptions_dict = dict()

            model.eval()
            with tqdm(total=val_num_batches) as pbar:
                with torch.no_grad():
                    for step, (img, gt_score_map, gt_geo_map, roi_mask, vertices, orig_sizes, labels, transcriptions, fnames) in enumerate(valid_loader):
                        pbar.set_description('[Valid {}]'.format(epoch + 1))

                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                        score_maps, geo_maps = extra_info['score_map'], extra_info['geo_map']
                        score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()

                        by_sample_bboxes = []
                        for i, (score_map, geo_map, orig_size, vertice, transcription, fname) in enumerate(zip(score_maps, geo_maps, orig_sizes, vertices, transcriptions, fnames)):
                            map_margin = int(abs(orig_size[0] - orig_size[1]) * 0.25 * image_size / max(orig_size))
                            if orig_size[0] > orig_size[1]:
                                score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
                            else:
                                score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

                            bboxes = get_bboxes(score_map, geo_map)
                            if bboxes is None:
                                bboxes = np.zeros((0, 4, 2), dtype=np.float32)
                            else:
                                bboxes = bboxes[:, :8].reshape(-1, 4, 2)

                            pred_bboxes_dict[fname] = bboxes
                            gt_bboxes_dict[fname] = vertice
                            transcriptions_dict[fname] = transcription

                        loss_val = loss.item()
                        if loss_val is not None:
                            val_epoch_loss += loss_val

                        pbar.update(1)
                        val_dict = {
                            'Cls loss': extra_info['cls_loss'],
                            'Angle loss': extra_info['angle_loss'],
                            'IoU loss': extra_info['iou_loss']
                        }
                        pbar.set_postfix(val_dict)

                        if extra_info['cls_loss'] is not None:
                            val_epoch_cls_loss += extra_info['cls_loss']
                            val_epoch_angle_loss += extra_info['angle_loss']
                            val_epoch_iou_loss += extra_info['iou_loss']
            resDict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict)
            f1_score = resDict['total']['hmean']
            print('[Valid {}]: f1_score : {:.4f} | precision : {:.4f} | recall : {:.4f}'.format(
                    epoch+1, resDict['total']['hmean'], resDict['total']['precision'], resDict['total']['recall']))

            wandb.log({"Val/F1": resDict['total']['hmean'],
                       "Val/Recall": resDict['total']['recall'],
                       "Val/Precision": resDict['total']['precision'],
                       "Val/Cls loss": val_epoch_cls_loss / val_num_batches,
                       "Val/Angle loss": val_epoch_angle_loss / val_num_batches,
                       "Val/IoU loss": val_epoch_iou_loss / val_num_batches,
                       #"Val/Loss": val_epoch_loss / val_num_batches,
                    })
            
            # Early Stopping + Update Best Epoch
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

        # Save checkpoint
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
