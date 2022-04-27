# coding: UTF-8
import time
import torch
import numpy as np
from muti_train_eval import train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, required=True, help='')
args = parser.parse_args()


if __name__ == '__main__':

    dataset = 'kubernetes'  # 数据集
    # dataset = 'minikube'
    # dataset = 'zephyr'
    # dataset = 'amphtml'
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
