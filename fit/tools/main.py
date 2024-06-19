import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
from meters import Meters
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from train import train
from transform import transform
from save import save_pic, save_params
from load import load
import torch
import numpy as np
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
import time
import logging

import argparse
import json

import matplotlib.pyplot as plt
import imageio, os


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Fit SMPL')
    parser.add_argument('--exp', dest='exp',
                        help='Define exp name',
                        default=time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())), type=str)
    parser.add_argument('--dataset_name', '-n', dest='dataset_name',
                        help='select dataset',
                        default='', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='path of dataset',
                        default=None, type=str)
    args = parser.parse_args()
    return args


def get_config(args):
    config_path = 'fit/configs/{}.json'.format(args.dataset_name)
    with open(config_path, 'r') as f:
        data = json.load(f)
    cfg = edict(data.copy())
    if not args.dataset_path == None:
        cfg.DATASET.PATH = args.dataset_path
    return cfg


def set_device(USE_GPU):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def get_logger(cur_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(os.path.join(cur_path, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(os.path.join(cur_path, 'tb'))

    return logger, writer


if __name__ == "__main__":
    args = parse_args()

    cur_path = os.path.join(os.getcwd(), 'exp', args.exp)
    assert not os.path.exists(cur_path), 'Duplicate exp name'
    os.mkdir(cur_path)

    cfg = get_config(args)
    json.dump(dict(cfg), open(os.path.join(cur_path, 'config.json'), 'w'))

    logger, writer = get_logger(cur_path)
    logger.info("Start print log")

    device = set_device(USE_GPU=cfg.USE_GPU)
    logger.info('using device: {}'.format(device))

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender=cfg.MODEL.GENDER,
        model_root='smplpytorch/native/models')
    
    meters = Meters()
    file_num = 0
    records = []
    for root, dirs, files in os.walk(cfg.DATASET.PATH):
        for file in sorted(files):
            record = {}
            # if not 'baseball_swing' in file:
            #     continue
            # if not file == 's010_driver02.npy':
            #     continue
            file_num += 1
            logger.info(
                'Processing file: {}    [{} / {}]'.format(file, file_num, len(files)))
            data = load(args.dataset_name, os.path.join(root, file))
            data[:,:,0] = data[:,:,0] - data[:,6:7,0]
            data[:,:,1] = data[:,:,1] - data[:,6:7,1]
            data[:,:,2] = data[:,:,2] - data[:,6:7,2]
            # print('Input: ', data.shape, data[200,:,2])
            target = torch.from_numpy(transform(args.dataset_name, data)).float()
            print('Target: ', target.shape, target[200,:,2])
            logger.info("target shape:{}".format(target.shape))
            res = train(smpl_layer, target,
                        logger, writer, device,
                        args, cfg, meters)
            meters.update_avg(meters.min_loss, k=target.shape[0])
            meters.reset_early_stop()
            logger.info("avg_loss:{:.4f}".format(meters.avg))
            record['filename'] = file
            record['loss'] = meters.avg
            records.append(record)
            save_params(res, smpl_layer, file, logger, args.dataset_name)
            save_pic(res, smpl_layer, file, logger, args.dataset_name, target)

            torch.cuda.empty_cache()
            images = []
            filenames = sorted(fn for fn in os.listdir(f'fit/output/{args.dataset_name}/picture/{file[:-4]}'))
            for filename in filenames:
                images.append(imageio.imread(f'fit/output/{args.dataset_name}/picture/{file[:-4]}/'+filename))
            imageio.mimsave(f'gif/{args.dataset_name}/{file[:-4]}.gif', images, duration=0.2)
            logger.info("Gifing finished!")
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(f'report/{args.dataset_name}', 'loss.csv'), index=False)
            
        logger.info("Fitting finished! Average loss: {:.9f}".format(meters.avg))

           
