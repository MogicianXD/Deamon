import collections
import importlib
import platform

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_provider.data_factory import data_provider, data_dict, resplit, get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Lead, Dataset_Recent
from exp.exp_basic import Exp_Basic
from exp.exp_main import Exp_Main
from exp.exp_online import Exp_Online
from models import LIFT, LIFT2, LIFT4
from peft import prompt
from util.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, load_model_compile, instance_norm
from util.metrics import metric, update_metrics, calculate_metrics

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch import optim

import os
import time
import copy

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Concept_Predict(Exp_Online):
    def __init__(self, args):
        super(Exp_Concept_Predict, self).__init__(args)
        if os.path.exists(args.concept_path):
            print('Loading concepts from', args.concept_path)
            concept = np.load(args.concept_path)
        else:
            concept = None
        self.label_position = 1
        self.wrap_data_kwargs.update(prompt_len=self.args.prompt_len, span=self.args.span, concept_path=args.concept_path,
                                     norm=self.args.concept_norm, bias=self.args.concept_bias,
                                     general_stat=self.args.general_stat, concept=concept,
                                     kind='whole' if self.args.whole_window else 'partial', penalty=self.args.penalty)

    def _get_data(self, flag, **kwargs):
        dataset, dataloader = super()._get_data(flag, **kwargs)
        if flag == 'online':
            if self.args.seq_len <= 336:
                dataloader.dataset.dataset.concept.cache = {k: v.to(self.device) for k, v in
                                                            dataloader.dataset.dataset.concept.cache.items()}
                print('Pin test data to GPU Mem')
        else:
            if flag in self.online_phases:
                dataloader.dataset.dataset.concept.cache = {k: v.to(self.device) for k, v in dataloader.dataset.dataset.concept.cache.items()}
                # dataloader.dataset.dataset.concept_label.cache = {k: v.to(self.device) for k, v in dataloader.dataset.dataset.concept_label.cache.items()}
            else:
                dataloader.dataset.concept.cache = {k: v.to(self.device) for k, v in dataloader.dataset.concept.cache.items()}
                # dataloader.dataset.concept_label.cache = {k: v.to(self.device) for k, v in dataloader.dataset.concept_label.cache.items()}
        return dataset, dataloader

    def _build_model(self, model=None, framework_class=None):
        model = importlib.import_module(f'models.concept_predictor.{self.args.model}').Model(self.args).float()
        return super()._build_model(model, framework_class)

    def _process_batch(self, batch):
        return [batch[-1].to(self.device)]
        # return ret[:-3] + ret[-2:]

    # def train_loss(self, criterion, batch, outputs):
    #     labels = batch[-3].to(self.device)
    #     return criterion(outputs, labels)

    def forward(self, batch):
        w = super().forward(batch)
        return self._infer(batch[0], w), w

    def _infer(self, X, w):
        X = X.permute(0, 2, 1)
        if self.args.concept_norm == 'last':
            last = X[..., [-1]]
            X = X - last
        elif self.args.concept_norm == 'instance':
            mean = X.mean(-1, keepdims=True)
            std = (((X - mean) ** 2).mean(-1, keepdims=True) + 1e-5) ** 0.5
            X = (X - mean) / std
        if self.args.concept_bias:
            X = torch.cat([torch.ones(*X.shape[:-1], 1, device=X.device), X], dim=-1)
        Y = X @ w.transpose(-1, -2)
        if self.args.concept_norm == 'last':
            Y = Y + last
        elif self.args.concept_norm == 'instance':
            Y = Y * std + mean
        return Y.transpose(-1, -2)

    def online(self, online_data=None, target_variate=None, phase='test'):
        self.phase = phase
        if hasattr(self.args, 'leakage') and self.args.leakage:
            return self.online_information_leakage(online_data, target_variate, phase)
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                      **self.wrap_data_kwargs)
            if self.args.seq_len <= 336:
                online_data.dataset.concept.cache = {k: v.to(self.device) for k, v in online_data.dataset.concept.cache.items()}
        online_loader_initial = get_dataloader(online_data.dataset, self.args, flag='online')
        online_loader = get_dataloader(online_data, self.args, flag='online')

        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}
        inference_num, batch_size = 0, online_loader_initial.batch_size
        self.model.eval()
        if self.args.do_predict:
            predictions = []
        with torch.no_grad():
            for i, current_data in enumerate(online_loader_initial):
                if (i + 1) * batch_size > online_data.gap:
                    if isinstance(current_data, tuple):
                        current_data = [b[:(online_data.gap - i * batch_size)] for b in current_data]
                    else:
                        current_data = current_data[:(online_data.gap - i * batch_size)]
                    inference_num += online_data.gap - i * batch_size
                else:
                    inference_num += batch_size
                outputs = self.forward(current_data)
                if self.args.do_predict:
                    predictions.append(outputs[1].detach().cpu().numpy())
                update_metrics(outputs[0], current_data[self.label_position], statistics, target_variate)
                if inference_num >= online_data.gap:
                    break

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if phase == 'test' and hasattr(self.args, 'debug') and self.args.debug:
            import tensorboardX as tensorboard
            import shutil
            log_dir = f'run/{self.args.online_method}_{self.args.dataset}_{self.args.seq_len}_{self.args.pred_len}_' \
                      f'{self.args.learning_rate}_{self.args.online_learning_rate}_{self.args.trigger_threshold}_' \
                      f'{self.args.tune_mode}_' \
                      f'{self.args.bottleneck_dim}_{self.args.concept_norm}_{self.args.penalty}_{self.args.comment}/' \
                      f'{time.strftime("%Y%m%d%H%M", time.localtime())}'
            print(log_dir)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            self.writer = tensorboard.SummaryWriter(log_dir=log_dir)

        if phase == 'test':
            online_loader = tqdm(online_loader, mininterval=10)
        for i, (recent_data, current_data) in enumerate(online_loader):
            self.model.train()
            if phase in self.online_phases:
                loss, _ = self._update_online(recent_data, criterion, model_optim, scaler)
            else:
                loss, _ = self._update(recent_data, criterion, model_optim, scaler)
            assert not torch.isnan(loss)
            self.model.eval()
            with torch.no_grad():
                outputs = self.forward(current_data)
                if self.args.do_predict:
                    predictions.append(outputs[1].detach().cpu().numpy())
                update_metrics(outputs[0], current_data[self.label_position], statistics, target_variate)

                if phase == 'test' and hasattr(self.args, 'debug') and self.args.debug:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    mse = F.mse_loss(outputs, current_data[self.label_position])
                    self.writer.add_scalar('Online/MSE', mse, i)
                    self.writer.add_scalar('Online/avg_MSE', statistics['MSE'] / statistics['total'], i)
                    # print('Online MSE: {:.2f}'.format(mse.item()))
                    # for j in range(current_data[self.label_position].shape[-1]):
                    #     self.writer.add_scalar(f'Online/x_{j}', current_data[self.label_position][0, 0, j], i)

        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        print('mse:{}, mae:{}'.format(mse, mae))
        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data

    def update_valid(self, valid_data=None):
        self.phase = 'online'
        if valid_data is None:
            valid_data = get_dataset(self.args, 'val', self.device, wrap_class=self.args.wrap_data_class,
                                     **self.wrap_data_kwargs)
            valid_data.concept.cache = {k: v.to(self.device) for k, v in valid_data.concept.cache.items()}
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.model.train()
        predictions = []
        for i, batch in enumerate(tqdm(valid_loader, mininterval=10)):
            _, outputs = self._update_online(batch, criterion, model_optim, scaler)
            if self.args.do_predict:
                predictions.append(outputs[1].detach().cpu().numpy())
        return predictions

    def inference_train(self, data=None):
        self.phase = 'infer'
        predictions = []
        if data is None:
            data = get_dataset(self.args, 'train', self.device, wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs, )
            data.concept.cache = {k: v.to(self.device) for k, v in data.concept.cache.items()}
        dataloader = get_dataloader(data, self.args, 'pred')
        for i, batch in enumerate(tqdm(dataloader, mininterval=10)):
            outputs = self.forward(batch)
            predictions.append(outputs[1].detach().cpu().numpy())
        return predictions
