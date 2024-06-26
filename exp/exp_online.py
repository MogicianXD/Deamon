import copy
from tqdm import tqdm

from data_provider.data_factory import data_provider, get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp.exp_main import Exp_Main
from models.OneNet import OneNet
from util.buffer import Buffer
from util.metrics import metric, update_metrics, calculate_metrics
import torch
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist

import os
import time

import warnings

from util.tools import test_params_flop

warnings.filterwarnings('ignore')

transformers = ['Autoformer', 'Transformer', 'Informer']

class Exp_Online(Exp_Main):
    def __init__(self, args):
        super().__init__(args)
        self.online_phases = ['test', 'online']
        self.wrap_data_kwargs.update(recent_num=1, gap=self.args.interval + self.args.pred_len - 1)

    def _get_data(self, flag, **kwargs):
        if flag in self.online_phases:
            if self.args.leakage:
                data_set = get_dataset(self.args, flag, self.device, wrap_class=self.args.wrap_data_class, **kwargs)
                data_loader = get_dataloader(data_set, self.args, 'online' if flag == 'test' else 'test')
            else:
                data_set = get_dataset(self.args, flag, self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                       **self.wrap_data_kwargs, **kwargs)
                data_loader = get_dataloader(data_set, self.args, 'online')
            return data_set, data_loader
        else:
            return super()._get_data(flag, **kwargs)

    def vali(self, vali_data, vali_loader, criterion):
        self.phase = 'val'
        if self.args.leakage or 'val' not in self.online_phases:
            mse = super().vali(vali_data, vali_loader, criterion)
        else:
            if self.args.local_rank <= 0:
                state_dict = copy.deepcopy(self.state_dict())
                mse = self.online(online_data=vali_data, target_variate=None, phase='val')[0]
                if self.args.local_rank == 0:
                    mse = torch.tensor(mse, device=self.device)
                self.load_state_dict(state_dict, strict=not (hasattr(self.args, 'freeze') and self.args.freeze))
            else:
                mse = torch.tensor(0, device=self.device)
            if self.args.local_rank >= 0:
                dist.all_reduce(mse, op=dist.ReduceOp.SUM)
                mse = mse.item()
        return mse

    def update_valid(self, valid_data=None):
        self.phase = 'online'
        if valid_data is None:
            valid_data = get_dataset(self.args, 'val', self.device, wrap_class=self.args.wrap_data_class,
                                     **self.wrap_data_kwargs)
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.model.train()
        predictions = []
        for i, batch in enumerate(tqdm(valid_loader, mininterval=10)):
            _, outputs = self._update_online(batch, criterion, model_optim, scaler)
            if self.args.do_predict:
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())
        return predictions

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        return self._update(batch, criterion, optimizer, scaler)

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        self.phase = phase
        if hasattr(self.args, 'leakage') and self.args.leakage:
            return self.online_information_leakage(online_data, target_variate, phase)
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                      **self.wrap_data_kwargs)
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
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)
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

        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=10)
        for i, (recent_data, current_data) in enumerate(online_loader):
            self.model.train()
            if phase in self.online_phases:
                loss, _ = self._update_online(recent_data, criterion, model_optim, scaler)
            else:
                loss, _ = self._update(recent_data, criterion, model_optim, scaler)
            # assert not torch.isnan(loss)
            self.model.eval()
            with torch.no_grad():
                outputs = self.forward(current_data)
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)

                if phase == 'test' and hasattr(self.args, 'debug') and self.args.debug:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    mse = F.mse_loss(outputs, current_data[self.label_position].to(self.device))
                    self.writer.add_scalar('Online/MSE', mse, i)
                    self.writer.add_scalar('Online/avg_MSE', statistics['MSE'] / statistics['total'], i)
                    # print('Online MSE: {:.2f}'.format(mse.item()))
                    # for j in range(current_data[self.label_position].shape[-1]):
                    #     self.writer.add_scalar(f'Online/x_{j}', current_data[self.label_position][0, 0, j], i)

        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        if phase == 'test':
            print('mse:{}, mae:{}'.format(mse, mae))
        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data

    def online_information_leakage(self, online_data=None, target_variate=None, phase='test'):
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class,
                                      **self.wrap_data_kwargs)
        online_loader_initial = get_dataloader(online_data, self.args, flag='online')

        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if phase == 'test':
            online_loader_initial = tqdm(online_loader_initial, mininterval=10)
        for i, current_data in enumerate(online_loader_initial):
            if phase in self.online_phases:
                loss, outputs = self._update_online(current_data, criterion, model_optim, scaler)
            else:
                loss, outputs = self._update(current_data, criterion, model_optim, scaler)
            with torch.no_grad():
                update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)

        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        if phase == 'test':
            print('mse:{}, mae:{}'.format(mse, mae))
        return mse, mae, online_data

    def analysis_online(self):
        online_data = get_dataset(self.args, 'test', self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                  **self.wrap_data_kwargs)
        online_loader = get_dataloader(online_data, self.args, flag='online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        times_update = []
        times_infer = []
        print('GPU Mem:', torch.cuda.max_memory_allocated())
        for i, (recent_data, current_data) in enumerate(online_loader):
            start_time = time.time()
            self.model.train()
            recent_data = [d.to(self.device) for d in recent_data]
            loss, _ = self._update_online(recent_data, criterion, model_optim, scaler)
            if i > 10:
                times_update.append(time.time() - start_time)
            self.model.eval()
            with torch.no_grad():
                start_time = time.time()
                current_data = [d.to(self.device) for d in current_data]
                self.forward(current_data)
            # if i == 0:
            #     print('New GPU Mem:', torch.cuda.memory_allocated())
            if i > 10:
                times_infer.append(time.time() - start_time)
            if i == 30:
                break
        print('Final GPU Mem:', torch.cuda.max_memory_allocated())
        times_update = (sum(times_update) - min(times_update) - max(times_update)) / (len(times_update) - 2)
        times_infer = (sum(times_infer) - min(times_infer) - max(times_infer)) / (len(times_infer) - 2)
        print('Update Time:', times_update)
        print('Infer Time:', times_infer)
        print('Latency:', times_update + times_infer)
        test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))


class Exp_ER(Exp_Online):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.online_phases = ['test', 'val', 'online']
        self.buffer = Buffer(500, self.device)
        self.count = 0

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(8)
            out = self.forward(buff[:-1])
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += 0.2 * criterion(out, buff[1])
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        loss, outputs = self._update(batch, criterion, optimizer, scaler=None)
        idx = self.count + torch.arange(batch[1].size(0)).to(self.device)
        self.count += batch[1].size(0)
        self.buffer.add_data(*(batch + (idx,)))
        return loss, outputs


class Exp_DERpp(Exp_ER):

    def train_loss(self, criterion, batch, outputs):
        loss = Exp_Online.train_loss(self, criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(8)
            out = self.forward(buff[:-1])
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += 0.2 * criterion(buff[-1], out)
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        loss, outputs = Exp_Online._update_online(self, batch, criterion, optimizer, scaler)
        self.count += batch[1].size(0)
        if isinstance(outputs, (tuple, list)):
            self.buffer.add_data(*(batch + [outputs[0]]))
        else:
            self.buffer.add_data(*(batch + [outputs]))
        return loss, outputs


class Exp_FSNet(Exp_Online):
    def __init__(self, args):
        super().__init__(args)
        self.online_phases = ['test', 'val', 'online']

    def _update(self, *args, **kwargs):
        ret = super()._update(*args, **kwargs)
        if hasattr(self.model, 'store_grad'):
            self.model.store_grad()
        return ret

    def vali(self, *args, **kwargs):
        if not hasattr(self.model, 'try_trigger_'):
            return super().vali(*args, **kwargs)
        else:
            self.model.try_trigger_(True)
            ret = super().vali(*args, **kwargs)
            self.model.try_trigger_(False)
            return ret

    def online(self, *args, **kwargs):
        if not hasattr(self.model, 'try_trigger_'):
            return super().online(*args, **kwargs)
        else:
            self.model.try_trigger_(True)
            ret = super().online(*args, **kwargs)
            self.model.try_trigger_(False)
            return ret


class Exp_OneNet(Exp_FSNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_w = optim.Adam([self.model.weight], lr=self.args.learning_rate_w)
        self.opt_bias = optim.Adam(self.model.decision.parameters(), lr=self.args.learning_rate_bias)
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        return super()._select_optimizer(filter_frozen, return_self, model=self.model.backbone)

    def state_dict(self, *args, **kwargs):
        destination = super().state_dict(*args, **kwargs)
        destination['opt_w'] = self.opt_w.state_dict()
        destination['opt_bias'] = self.opt_bias.state_dict()
        return destination

    # def load_state_dict(self, state_dict, model=None):
    #     self.model.bias.data = state_dict['model']['bias']
    #     return super().load_state_dict(state_dict, model)

    def _build_model(self, model=None, framework_class=None):
        return super()._build_model(model, framework_class=OneNet)

    def train_loss(self, criterion, batch, outputs):
        return super().train_loss(criterion, batch, outputs[1]) + super().train_loss(criterion, batch, outputs[2])

    def vali(self, vali_data, vali_loader, criterion):
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)
        ret = super().vali(vali_data, vali_loader, criterion)
        self.phase = None
        return ret

    def update_valid(self, valid_data=None):
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)
        return super().update_valid(valid_data)

    def forward(self, batch):
        b, t, d = batch[1].shape
        if hasattr(self, 'phase') and self.phase in self.online_phases:
            weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
            bias = self.bias.view(-1, 1, d)
            loss1 = F.sigmoid(weight + bias.repeat(1, t, 1)).view(b, t, d)
        else:
            loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
            loss1 = loss1.repeat(b, t, 1)
        batch = batch + [loss1, 1 - loss1]
        return super().forward(batch)

    def _update(self, batch, criterion, optimizer, scaler=None):
        batch_y = batch[1]
        b, t, d = batch_y.shape

        loss, (outputs, y1, y2) = super()._update(batch, criterion, optimizer, scaler)

        loss_w = criterion(outputs, batch_y)
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()

        y1_w, y2_w = y1.detach(), y2.detach()
        true_w = batch_y.detach()
        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, true_w], dim=1)
        bias = self.model.decision(inputs_decision.permute(0, 2, 1)).view(b, 1, -1)
        weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
        loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
        loss2 = 1 - loss1
        loss_bias = criterion(loss1 * y1_w + loss2 * y2_w, true_w)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        return loss / 2, outputs

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        batch_y = batch[1]
        b, t, d = batch_y.shape

        loss, (outputs, y1, y2) = super()._update(batch, criterion, optimizer, scaler)

        y1_w, y2_w = y1.detach(), y2.detach()
        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1).repeat(b, t, 1)
        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, batch_y], dim=1)
        self.bias = self.model.decision(inputs_decision.permute(0, 2, 1))
        weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
        bias = self.bias.view(b, 1, -1)
        loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
        loss2 = 1 - loss1

        outputs_bias = loss1 * y1_w + loss2 * y2_w
        loss_bias = criterion(outputs_bias, batch_y)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        loss_w = criterion(loss1 * y1_w + (1 - loss1) * y2_w, batch_y)
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()
        return loss / 2, outputs

