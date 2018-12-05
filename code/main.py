from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import pickle

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


from miscc.config import cfg, cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/birds_proGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    #parser.add_argument('--config_key',dest='config_key', type=str, help='configuration name', default = 'finegan_birds')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if cfg.TRAIN.FLAG:
        print('Using config:')
        pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 45   # Change this to have different random seed during evaluation

    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    
    # Evaluation part
    if not cfg.TRAIN.FLAG:
        from trainer import FineGAN_evaluator as evaluator
        algo = evaluator()
        algo.evaluate_finegan()

    # Training part
    else:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = '../output/%s_%s' % \
            (cfg.DATASET_NAME, timestamp)
        pkl_filename = 'cfg.pickle'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, pkl_filename), 'wb') as pk:
            pickle.dump(cfg, pk, protocol=pickle.HIGHEST_PROTOCOL)

        bshuffle = True

        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
        image_transform = transforms.Compose([
            transforms.Scale(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])


        from datasets import Dataset
        dataset = Dataset(cfg.DATA_DIR,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        assert dataset
        num_gpu = len(cfg.GPU_ID.split(','))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))


        from trainer import FineGAN_trainer as trainer
        algo = trainer(output_dir, dataloader, imsize)

        start_t = time.time()
        algo.train()
        end_t = time.time()
        print('Total time for training:', end_t - start_t)
