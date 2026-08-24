import os
import random
import argparse

import numpy as np
import torch
from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config):
    set_seed(config.seed)
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'visualize':
        solver.visualize_event_case()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='STOCK')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize'])
    parser.add_argument('--data_path', type=str, default='SP500')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--stride', type=int, default=1)

    parser.add_argument('--date_col', type=str, default='date')
    parser.add_argument('--ticker_col', type=str, default='ticker')
    parser.add_argument('--open_col', type=str, default='open')
    parser.add_argument('--high_col', type=str, default='high')
    parser.add_argument('--low_col', type=str, default='low')
    parser.add_argument('--close_col', type=str, default='close')
    parser.add_argument('--volume_col', type=str, default='volume')
    parser.add_argument('--tickers', type=str, default=None)
    parser.add_argument('--features', type=str, default='log_return_1d,return_5d,return_20d,volume_z,abs_return,squared_return,rolling_vol_5,rolling_vol_20,vol_ratio_5_20,gap,high_low_range')
    parser.add_argument('--z_state_features', type=str, default='log_return_1d,abs_return,volume_z,rolling_vol_5,rolling_vol_20,vol_ratio_5_20')
    parser.add_argument('--state_projection_dim', type=int, default=0)
    parser.add_argument('--token_embedding', type=str, default='conv3_circular',
                        choices=['conv3_circular', 'linear', 'linear_norm', 'linear_residual_mlp',
                                 'grouped'])
    parser.add_argument('--use_positional_embedding', type=str2bool, default=True)
    parser.add_argument('--position_sigma', type=float, default=0.0)
    parser.add_argument('--volume_window', type=int, default=60)
    parser.add_argument('--label_window', type=int, default=60)
    parser.add_argument('--feature_cache_dir', type=str, default=None)
    parser.add_argument('--train_start', type=str, default=None)
    parser.add_argument('--train_end', type=str, default=None)
    parser.add_argument('--val_start', type=str, default=None)
    parser.add_argument('--val_end', type=str, default=None)
    parser.add_argument('--test_start', type=str, default=None)
    parser.add_argument('--test_end', type=str, default=None)

    parser.add_argument('--prior_type', type=str, default='time', choices=['time', 'time_state', 'state'])
    parser.add_argument('--return_loss_weight', type=float, default=0.0)
    parser.add_argument('--use_return_nll', type=str2bool, default=False)
    parser.add_argument('--nll_weight', type=float, default=0.0)
    parser.add_argument('--score_type', type=str, default='original',
                        choices=['original', 'return_recon', 'return_nll', 'feature_weighted',
                                 'calibrated_components'])
    parser.add_argument('--score_normalization', type=str, default='none',
                        choices=['none', 'ticker_percentile'])
    parser.add_argument('--feature_weights', type=str, default=None)
    parser.add_argument('--reconstruction_features', type=str, default=None)
    parser.add_argument('--calibrated_score_features', type=str, default='log_return_1d,volume_z,gap')
    parser.add_argument('--score_aggregation', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--label_type', type=str, default='absolute',
                        choices=['absolute', 'absolute_z5', 'return', 'volume', 'contextual',
                                 'return_volume_3std'])
    parser.add_argument('--threshold_method', type=str, default='percentile', choices=['percentile', 'best_f1'])
    parser.add_argument('--threshold_percentile', type=float, default=99.0)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--event_tolerance', type=int, default=1)

    parser.add_argument('--visualize_ticker', type=str, default=None)
    parser.add_argument('--event_date', type=str, default=None)
    parser.add_argument('--auto_case_ticker', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='figures')
    parser.add_argument('--run_root', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--plot_layer', type=int, default=0)
    parser.add_argument('--plot_head', type=str, default='average')

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
