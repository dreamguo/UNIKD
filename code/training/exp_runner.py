import os
import sys
import ipdb
import torch
import argparse
sys.path.append('../code')
from training.monosdf_incre_train import MonoSDFIncreTrainRunner
import datetime


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepochs', type=int, default=1000, help='number of epochs to train for, 10000-ICL, 2000-replica')
    parser.add_argument('--tensorboard_freq', type=int, default=100)
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')

    # use for ICL dataset
    parser.add_argument("--gt_depth", type=int, default=0)

    # test
    parser.add_argument("--infer", type=int, default=0)
    parser.add_argument("--plot_only_mesh", type=int, default=0)
    parser.add_argument("--varify_filter", type=int, default=0, help='when ppose==vae can be 1')

    # train
    parser.add_argument("--training_type", type=str, default='', required=True,
                        help='ours / baseline / mas / afc / podnet / keyf / no_init / no_filter / batch_train / save / incre_train / block')
    parser.add_argument("--incre_timestamp", type=str, default='')
    parser.add_argument("--incremental", type=int, default=10)
    parser.add_argument("--teacher_freq", type=int, default=2)
    parser.add_argument("--block", type=str, default='')
    parser.add_argument("--accumulate_data", type=int, default=0)

    # disloss
    parser.add_argument("--disloss", type=int, default=1)
    parser.add_argument("--ppose", type=str, default='random', help='vae / random / save')
    # use for ppose == random
    parser.add_argument("--use_uncertainty_loss", type=int, default=0)
    parser.add_argument("--uncertainty_weight", type=float, default=0.01)
    parser.add_argument("--uncertain_C_thresh", type=float, default=999999)

    # baselines
    parser.add_argument("--use_mas", type=int, default=0)  # mas baseline
    parser.add_argument("--use_podnet", type=int, default=0)  # podnet baseline
    parser.add_argument("--keyf_num", type=int, default=0)  # save-rgb baseline
    parser.add_argument("--packnet", type=int, default=0)  # packnet baseline

    parser.add_argument("--cross", type=int, default=0)

    opt = parser.parse_args()

    if opt.incre_timestamp == '':
        opt.incre_timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())

    if opt.training_type == 'ours':
        opt.disloss = 1
        opt.ppose = 'random'
        opt.use_uncertainty_loss = 1
        opt.uncertainty_weight = 0.01
        opt.uncertain_C_thresh = 0.02
    elif opt.training_type == 'no_init':  # no_init & no_filter
        opt.disloss = 1
        opt.ppose = 'random'
        opt.use_uncertainty_loss = 0
    elif opt.training_type == 'save':
        opt.disloss = 1
        opt.ppose = 'save'
    elif opt.training_type == 'baseline':
        opt.disloss = 0
    elif opt.training_type == 'mas':
        opt.disloss = 0
        opt.use_mas = 1
        opt.mas_lambda = 1
    elif opt.training_type == 'afc':
        opt.disloss = 1
        opt.use_mas = 1
        opt.mas_lambda = 1
    elif opt.training_type == 'podnet':
        opt.disloss = 1
        opt.ppose = 'random'
        opt.use_uncertainty_loss = 0
        opt.use_podnet = 1
    elif opt.training_type == 'keyf':
        opt.disloss = 0
        if opt.keyf_num == 0:
            opt.keyf_num = 10
    elif opt.training_type == 'packnet':
        opt.disloss = 0
        opt.packnet = 1
    elif opt.training_type == 'no_filter':
        opt.disloss = 1
        opt.ppose = 'random'
        opt.use_uncertainty_loss = 0
    elif opt.training_type == 'batch_train':
        opt.incremental = 0
        opt.disloss = 0
    elif opt.training_type == 'incre_train':
        opt.cross = 0
        opt.accumulate_data = 1
        opt.disloss = 0
    elif opt.training_type == 'block':
        opt.incremental = 0
        opt.disloss = 0
    else:
        assert False, "Invalid training_type"

    opt.step_i = 0
    if opt.incremental != 0:
        opt.total_nepochs = opt.incremental * opt.nepochs
        step_epoch = opt.nepochs
        for step_i in range(opt.incremental):
            opt.step_i = step_i
            if opt.step_i == 0:
                opt.nepochs = step_epoch
            else:
                opt.nepochs += step_epoch
            trainrunner = MonoSDFIncreTrainRunner(opt)
            if opt.infer:
                trainrunner.infer()
                continue
            if not opt.varify_filter and trainrunner.start_epoch < trainrunner.nepochs:
                trainrunner.run()
            else:
                print('---------------------------------------------Skip training for step ', opt.step_i, '---------------------------------------------')
    else:
        opt.total_nepochs = opt.nepochs
        trainrunner = MonoSDFIncreTrainRunner(opt)
        if opt.infer:
            trainrunner.infer()
        else:
            trainrunner.run()
