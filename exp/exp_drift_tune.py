import copy

import torch
from torch import optim
from tqdm import tqdm

from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Concept_Pred
from exp import Exp_Online
import peft
from peft.hyper.base import Adaptation
from peft import drift_tuning, drift_tune
from peft.hyper.factory import Adapters
from util.functional import get_concept


class Exp_Drift_Tune(Exp_Online):
    def __init__(self, args):
        args = copy.deepcopy(args)
        args.pretrain = True
        args.freeze = True
        args.merge_weights = 1
        # if args.model != 'GPT4TS':
        #     args.tune_mode = 'lora_up'
        # else:
        #     args.concept_norm = 'instance'
        if args.tune_mode in ['only_up', 'ssf']:
            args.more_bias = True
        elif args.tune_mode == 'lora_up':
            args.more_bias = False
        super(Exp_Drift_Tune, self).__init__(args)
        if Dataset_Concept_Pred not in self.args.wrap_data_class:
            self.args.wrap_data_class.append(Dataset_Concept_Pred)
        self.online_phases = ['test', 'online']
        # if args.model != 'GPT4TS':
        self.online_phases += ['val']
        self.wrap_data_kwargs.update(prompt_len=self.args.prompt_len, span=self.args.span,
                                     norm=self.args.concept_norm, bias=self.args.concept_bias,
                                     general_stat=self.args.general_stat,
                                     kind='whole' if self.args.whole_window else 'partial', penalty=self.args.penalty)
        self.mean_dim = args.seq_len if self.args.general_stat else args.enc_in

    @property
    def _model(self) -> drift_tune.DriftTune:
        if self.args.local_rank >= 0:
            return self.model.module
        return self.model

    def _get_data(self, flag, **kwargs):
        # if flag == 'train':
        #     dataset = get_dataset(self.args, flag, self.device, wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs, **kwargs)
        #     dataloader = get_dataloader(dataset, self.args, 'online')
        # else:
        dataset, dataloader = super()._get_data(flag, **kwargs)
        flag_to_cuda = self.args.seq_len <= 60
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
            self._model.recent_concept = torch.cat([dataset.cache['Y|X'].mean(0).view(1, -1),
                                                    dataset.cache[dataset.return_key[1]].mean(0, keepdim=True)], -1).to(
                self.device)

        elif flag == 'val' and flag_to_cuda:
            if 'val' in self.online_phases:
                dataloader.dataset.dataset.cache = {k: v.to(self.device) for k, v in
                                                    dataloader.dataset.dataset.cache.items()}
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
        return ret

    def update_valid(self, valid_data=None):
        self._model.flag_reuse = self.args.reuse
        self._model.gamma = self.args.ema
        ret = super().update_valid(valid_data)
        self.model_optim.zero_grad()
        self._model.freeze_encoder(True)
        trainable_params = sum([param.nelement() if param.requires_grad else 0 for param in self._model.parameters()])
        print(f'Trainable Params: {trainable_params}', '({:.1f}%)'.format(trainable_params / self.model_params * 100))
        return ret

    def _build_model(self, model=None, framework_class=None):
        return super()._build_model(model,
                                    framework_class=drift_tuning.DriftTune if self.args.chunk else drift_tune.DriftTune)

    def _update(self, batch, criterion, optimizer, scaler=None):
        self._model.flag_update = True
        self._model.mode(Tuners.MODE_NOW)
        loss, outputs = super()._update(batch, criterion, optimizer, scaler)
        optimizer.zero_grad()
        self._model.mode(Tuners.MODE_LAST)
        super()._update(batch, criterion, optimizer, scaler)
        if hasattr(self, 'phase') and self.phase in self.online_phases:
            self._model.recent_concept = batch[-2].mean(0, keepdims=True).to(self.device)
        else:
            X, Y = [d.permute(0, 2, 1).to(self.device) for d in batch[:2]]
            concept_YX = get_concept(X, Y, self.args.concept_norm, self.args.penalty, self.args.concept_bias)
            self._model.recent_concept = torch.cat(
                [concept_YX.reshape(1, -1), batch[-2][:, -self.mean_dim:].mean(0, keepdims=True).to(self.device)], -1)
        self._model.flag_update = False
        return loss, outputs

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        self._model.flag_online_learning = True
        if self.phase == 'online' or not self.args.freeze_delta:
            loss, outputs = self._update(batch, criterion, optimizer, scaler)
        else:
            self._model.mode(Tuners.MODE_LAST)
            loss, outputs = super()._update(batch, criterion, optimizer, scaler)
        self._model.mode(Tuners.MODE_NOW)
        if self._model.gamma > 0:
            # if self.phase == 'test' and self.args.freeze_delta:
            #     for adapter in self._model.tuners.adapters.values():
            #         adapter.update_ema(adapter.last_adaptation, ema=self._model.gamma)
            # else:
            self._model.generate_adaptation(batch[-1].to(self.device), resume=True,
                                            save=self._model.flag_reuse, update_lora_cache=False)
        self._model.recent_concept = batch[-2].to(self.device)
        self._model.flag_online_learning = False
        return loss, outputs
