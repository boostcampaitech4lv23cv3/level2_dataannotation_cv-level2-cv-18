import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm
import lanms
from detect import detect


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=20)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='public'):
    try:
        checkpoint = torch.load(ckpt_fpath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []
    
    input_size_list = [1024, 2048]
    weight_list = [0.9, 0.8]
    # images = []
    for image_fpath in tqdm(glob(osp.join(data_dir, '{}/*'.format(split)))):
        image_fnames.append(osp.basename(image_fpath))

        image = cv2.imread(image_fpath)[:, :, ::-1]
        temp_bboxes = [] ###
        for size, weight in zip(input_size_list, weight_list):
            temp = detect(model, [image], size)
            for t in temp:
                bboxes = t.reshape(-1, 8)
                for bbox in bboxes:
                    temp_bboxes.append(np.append(bbox, weight))
        temp_bboxes = np.array(temp_bboxes)
        temp_bboxes = lanms.merge_quadrangle_n9(temp_bboxes, 0.2)
        if len(temp_bboxes) == 0:
            bboxes = np.zeros((0, 4, 2), dtype=np.float32)
        else:
            bboxes = temp_bboxes[:, :8].reshape(-1, 4, 2)
        by_sample_bboxes.extend([bboxes])
    #     images.append(cv2.imread(image_fpath)[:, :, ::-1])
    #     if len(images) == batch_size:
    #         temp_bboxes = [] ###
    #         for size, weight in zip(input_size_list, weight_list):
    #             temp = detect(model, images, size)
    #             for t in temp:
    #                 bboxes = t.reshape(-1, 8)
    #                 for bbox in bboxes:
    #                     temp_bboxes.append(np.append(bbox, weight))
    #         temp_bboxes = np.array(temp_bboxes)
    #         temp_bboxes = lanms.merge_quadrangle_n9(temp_bboxes, 0.2)
    #         if temp_bboxes is None:
    #             bboxes = np.zeros((0, 4, 2), dtype=np.float32)
    #         else:
    #             bboxes = temp_bboxes[:, :8].reshape(-1, 4, 2)
    #         by_sample_bboxes.extend(bboxes)
    #         images = []

    # if len(images):
    #     temp_bboxes = [] ###
    #     for size, weight in zip(input_size_list, weight_list):
    #         temp = detect(model, images, size)
    #         for t in temp:
    #             bboxes = t.reshape(-1, 8)
    #             for bbox in bboxes:
    #                 temp_bboxes.append(np.append(bbox, weight))
    #     temp_bboxes = np.array(temp_bboxes)
    #     temp_bboxes = lanms.merge_quadrangle_n9(temp_bboxes, 0.2)
    #     if temp_bboxes is None:
    #         bboxes = np.zeros((0, 4, 2), dtype=np.float32)
    #     else:
    #         bboxes = temp_bboxes[:, :8].reshape(-1, 4, 2)
    #     by_sample_bboxes.extend(bboxes)

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    for split in ['public', 'private']:
        print('Split: {}'.format(split))
        split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                    args.batch_size, split=split)
        ufo_result['images'].update(split_result['images'])

    output_fname = 'output.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
