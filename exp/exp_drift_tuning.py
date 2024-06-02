import copy
import math
import os.path
import platform

import numpy as np
import torch

from data_provider.data_loader import Dataset_Concept_Pred
from exp import Exp_Online
from peft import drift_tuning
from util.functional import get_concept
import gc

class LossStat:
    def __init__(self, ema=1):
        self.mean = 0
        self.S = 0
        self.num = 0
        self.ema = ema
        self.record_stat = False
        self.ready = False
        self.update_flag = False

    def update_(self, loss):
        self.num += 1
        mean_pre = self.mean
        if self.ema == 1:
            self.mean = (self.mean * (self.num - 1) + loss) / self.num
            self.S += (loss - mean_pre) * (loss - self.mean)
        else:
            self.mean = ((1 - self.ema) * self.mean * (self.num - 1) + loss * self.ema) / self.num
            self.S = self.ema * self.S + (1 - self.ema) * (loss - mean_pre) * (loss - self.mean)

    @property
    def std(self):
        return math.sqrt(self.S / self.num)


class Exp_Drift_Tuning(Exp_Online):
    def __init__(self, args):
        args = copy.deepcopy(args)
        args.pretrain = True
        args.freeze = True
        args.merge_weights = 1
        # if args.model != 'GPT4TS':
        #     args.tune_mode = 'lora_up'
        # else:
        #     args.concept_norm = 'instance'
        if args.lora_rank == 0 and args.tune_mode == 'lora_up':
            args.tune_mode = 'up'
        if args.tune_mode in ['only_up', 'up', 'ssf', 'adapter', 'mix']:
            args.more_bias = True
        elif args.tune_mode in ['lora_up']:
            args.more_bias = False
        super(Exp_Drift_Tuning, self).__init__(args)
        if Dataset_Concept_Pred not in self.args.wrap_data_class:
            self.args.wrap_data_class.append(Dataset_Concept_Pred)
        self.online_phases = ['test', 'online']
        if args.model != 'GPT4TS':
            self.online_phases += ['val']

        if os.path.exists(args.concept_path):
            print('Loading concepts from', args.concept_path)
            concept = np.load(args.concept_path)
        else:
            concept = None
        self.wrap_data_kwargs.update(prompt_len=self.args.prompt_len, span=self.args.span,
                                     norm=self.args.concept_norm, bias=self.args.concept_bias, use_mean=self.args.use_mean,
                                     general_stat=self.args.general_stat, concept=concept, concept_path=args.concept_path,
                                     kind='whole' if self.args.whole_window else 'partial', penalty=self.args.penalty)
        self.mean_dim = args.seq_len if self.args.general_stat else args.enc_in
        self.update_cnt = 0
        # self._model.prune = False
        self.loss_stat = LossStat()

    @property
    def _model(self) -> drift_tuning.DriftTune:
        if self.args.local_rank >= 0:
            return self.model.module
        return self.model

    def _get_data(self, flag, **kwargs):
        # if flag == 'train':
        #     dataset = get_dataset(self.args, flag, self.device, wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs, **kwargs)
        #     dataloader = get_dataloader(dataset, self.args, 'online')
        # else:
        dataset, dataloader = super()._get_data(flag, **kwargs)
        flag_to_cuda = not (self.args.dataset in ['ECL', 'Traffic'] and self.args.seq_len >= 336)
        # flag_to_cuda = self.args.seq_len <= 60
        if flag == 'online':
            if self.args.seq_len <= 336:
                dataloader.dataset.dataset.cache = {k: v.to(self.device) for k, v in dataloader.dataset.dataset.cache.items()}
        if flag == 'train':
            if flag_to_cuda:
                dataloader.dataset.cache = {k: v.to(self.device) for k, v in dataloader.dataset.cache.items()}
            # concept = torch.cat([dataset.cache['Y|X'].view(len(dataset.cache['Y|X']), -1),
            #                      dataset.cache['mu']], -1)
            # norm2 = concept.pow(2).sum(dim=-1, keepdim=True)
            # self._model.max_norm = torch.addmm(norm2.transpose(0, 1),
            #                                   concept, concept.transpose(0, 1), alpha=-2).add_(norm2).max().sqrt().item()
            # norm = (dataset.cache['mu'][self.args.pred_len:] - dataset.cache['mu'][:-self.args.pred_len]).pow(2).sum(-1)
            # norm += (dataset.cache['pred'][self.args.pred_len:] - dataset.cache['Y|X'][:-self.args.pred_len]).pow(2).sum((-1, -2))
            # self._model.max_norm = max(1, norm.max().sqrt().item())
            # print('Max norm', self._model.max_norm)
            if self.args.use_mean:
                self._model.recent_concept = torch.cat([dataset.cache['Y|X'].mean(0).view(1, -1),
                                                      dataset.cache[dataset.return_key[1]].mean(0, keepdim=True)], -1).to(self.device)
            else:
                self._model.recent_concept = dataset.cache['Y|X'].mean(0).view(1, -1).to(self.device)

        elif flag == 'val':
            if self.args.do_valid:
                del self.wrap_data_kwargs['prediction']
                gc.collect()
            if flag_to_cuda:
                if 'val' in self.online_phases:
                    dataloader.dataset.dataset.cache = {k: v.to(self.device) for k, v in dataloader.dataset.dataset.cache.items()}
                else:
                    dataloader.dataset.cache = {k: v.to(self.device) for k, v in dataloader.dataset.cache.items()}
        return dataset, dataloader

    def online(self, *args, **kwargs):
        self.model_optim.zero_grad()
        self._model.freeze_encoder(True)
        self._model.gamma = self.args.ema
        ret = super().online(*args, **kwargs)
        self._model.gamma = 0
        self._model.freeze_encoder(False)
        # print('Update Count', self.update_cnt)
        return ret

    def update_valid(self, valid_data=None, valid_dataloader=None):
        self.loss_stat.record_stat = self.args.dynamic_freeze
        self._model.flag_reuse = self.args.reuse
        self._model.gamma = self.args.ema
        # if self.args.dynamic_freeze:
        #     self._model.freeze_encoder(True)
        ret = super().update_valid(valid_data)
        # print('Update Count', self.update_cnt)
        self.model_optim.zero_grad()
        self._model.freeze_encoder(True)
        print(self._model.ema)
        trainable_params = sum([param.nelement() if param.requires_grad else 0 for param in self._model.parameters()])
        print(f'Trainable Params: {trainable_params}', '({:.1f}%)'.format(trainable_params / self.model_params * 100))
        self.loss_stat.ready = self.args.dynamic_freeze
        return ret

    def _build_model(self, model=None, framework_class=None):
        model = super()._build_model(model, framework_class=drift_tuning.DriftTune)
        return model

    def _update(self, batch, criterion, optimizer, scaler=None):
        self._model.flag_update = True
        if self.args.anneal:
            self._model.adapters.temperature.data = torch.exp(torch.tensor(max(0.1, -1e-3 * self.update_cnt),
                                                                           device=self._model.adapters.temperature.device))
        loss, outputs = super()._update(batch, criterion, optimizer, scaler)
        self.update_cnt += 1
        # if self._model.flag_online_learning:
        #     self._model.recompute(drift, learning_rate=optimizer.param_groups[-1]['lr'])
        if self.args.batch_size == 1 or hasattr(self, 'phase') and self.phase in self.online_phases:
            self._model.recent_concept = batch[-2].mean(0, keepdims=True).to(self.device)
        else:
            X, Y = [d.permute(0, 2, 1).to(self.device) for d in batch[:2]]
            concept_YX = get_concept(X, Y, self.args.concept_norm, self.args.penalty, self.args.concept_bias)
            if self.args.use_mean:
                self._model.recent_concept = torch.cat([concept_YX.reshape(1, -1), batch[-2][:, -self.mean_dim:].mean(0, keepdims=True).to(self.device)], -1)
            else:
                self._model.recent_concept = concept_YX.reshape(1, -1)
        self._model.flag_update = False
        # self.update_cnt += 1
        # if self.update_cnt > 10:
        #     self._model.prune = self.args.prune
        return loss, outputs

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        if self.loss_stat.ready:
            if self.loss_stat.num > 0:
                self.loss_stat.update_flag = loss >= self.loss_stat.mean + self.loss_stat.std
                if not self.loss_stat.update_flag:
                    self._model.freeze_encoder(True)
        if self.loss_stat.record_stat:
            self.loss_stat.update_(loss.item())
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        # if self._model.flag_reuse:
        self._model.flag_online_learning = True
        # if self.args.debug:
        #     print('Finetune Loss: ', end='')
        if self.loss_stat.ready:
            self._model.freeze_encoder(False, include_prune=False)
        loss, outputs = super()._update(batch, criterion, optimizer, scaler)
        if self.loss_stat.update_flag:
            self._model.freeze_encoder(True)
        # if self._model.gamma > 0:
        #     if self.phase == 'test' and self.args.freeze_delta:
        #         for adapter in self._model.adapters.adapters.values():
        #             adapter.update_ema(adapter.last_adaptation, ema=self._model.gamma)
        #     else:
        #         self._model.generate_adaptation(batch[-1].to(self.device), resume=True,
        #                                         save=self._model.flag_reuse, update_lora_cache=False)
        self._model.recent_concept = batch[-2].to(self.device)
        if self._model.flag_reuse:
            sim_K, indices, is_new_concept = self._model.eval_concept(self._model.recent_concept, include_recent=False)
            self._model.memorize_concept(self._model.recent_concept, is_new_concept, indices, learning_rate=0, recompute=False)
            print(len(self._model.concept_pool))
                      # , end='\t' if self.phase == 'test' else '\n')
            # else:
            #     print()
        self._model.flag_online_learning = False
        return loss, outputs
