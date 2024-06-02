import argparse
import copy
import datetime
import os
import time
from data_provider import data_loader

from exp import exp_online
from exp.exp_concept import Exp_Concept_Predict
from exp.exp_online import Exp_Online

ds = time.strftime("%Y%m%d", time.localtime())
dh = time.strftime("%Y%m%d%H", time.localtime())
cur_sec = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(cur_sec)

from pprint import pprint

import torch

import settings
import random
import numpy as np

from settings import data_settings


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--train_only', action='store_true', default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--wo_test', action='store_true', default=False, help='only valid, not test')
# parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--only_test', action='store_true', default=False)
parser.add_argument('--do_valid', action='store_true', default=False)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')
parser.add_argument('--override_hyper', action='store_true', default=True, help='Override hyperparams by setting.py')
parser.add_argument('--compile', action='store_true', default=False, help='Compile the model by Pytorch 2.0')
parser.add_argument('--reduce_bs', type=str_to_bool, default=False, help='Override batch_size in hyperparams by setting.py')
parser.add_argument('--extend', action='store_true', default=False)
parser.add_argument('--normalization', type=str, default=None)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# online
parser.add_argument('--online_method', type=str, default=False)
parser.add_argument('--online_learning_rate', type=float, default=None)
parser.add_argument('--border_type', type=str, default='')
parser.add_argument('--save_opt', action='store_true', default=True)
parser.add_argument('--leakage', action='store_true', default=False)
parser.add_argument('--interval', type=int, default=1, help='online task interval')


# DishTS
parser.add_argument('--alpha', type=float, default=1)

# data loader
# parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--dataset', type=str, default='ETTh1', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--wrap_data_class', type=list, default=[])

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# LAL
parser.add_argument('--leader_num', type=int, default=4, help='# of leaders')
parser.add_argument('--leader_select_num', type=int, default=1, help='# of leaders')
parser.add_argument('--trunc_tail', type=int, default=0, help='truncate out the last ones of correlations')
parser.add_argument('--prefetch_path', type=str, default='./prefetch/', help='location of prefetch files')
parser.add_argument('--tag', type=str, default='_max')
parser.add_argument('--prefetch_batch_size', type=int, default=16, help='prefetch_batch_size')
parser.add_argument('--variable_batch_size', type=int, default=32, help='variable_batch_size')
parser.add_argument('--max_leader_num', type=int, default=32, help='max # of leaders')
parser.add_argument('--predefined_search_space', action='store_true', default=False)
parser.add_argument('--test_speed', action='store_true', default=False)
parser.add_argument('--mix', action='store_true', default=True)
parser.add_argument('--wo_infl', action='store_true', default=False)
parser.add_argument('--wo_diff', action='store_true', default=False)
parser.add_argument('--wo_filter', action='store_true', default=False)
parser.add_argument('--wo_softmax', action='store_true', default=False)
parser.add_argument('--masked_corr', action='store_true', default=False)
parser.add_argument('--efficient', type=str_to_bool, default=True)
parser.add_argument('--pin_gpu', type=str_to_bool, default=True)
parser.add_argument('--univariate', action='store_true', default=False)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--freeze', action='store_true', default=False)
parser.add_argument('--lift', action='store_true', default=False)
parser.add_argument('--seg', action='store_true', default=False)
parser.add_argument('--decom', action='store_true', default=False)
parser.add_argument('--min_lag', type=int, default=1, help='lower bound of lag')
parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')
parser.add_argument('--state_num', type=int, default=8, help='# of MOE')
# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
# parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
# parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--output_enc', action='store_true', help='whether to output embedding from encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# Crossformer
parser.add_argument('--seg_len', type=int, default=24, help='segment length (L_seg)')
parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
parser.add_argument('--num_routers', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')


# GPT4TS
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--tmax', type=int, default=10)
parser.add_argument('--patch_size', type=int, default=16)

# OneNet
parser.add_argument('--learning_rate_w', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--learning_rate_bias', type=float, default=0.001, help='optimizer learning rate')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--begin_valid_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--warmup_epochs', type=int, default=8)

# GPU
parser.add_argument('--use_gpu', type=str_to_bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)

# concept
parser.add_argument('--whole_window', type=str_to_bool, default=False)
parser.add_argument('--prompt_len', type=int, default=14)
parser.add_argument('--y_emb_dim', type=int, default=8)
parser.add_argument('--span', type=int, default=24)
parser.add_argument('--penalty', type=float, default=1.0)
parser.add_argument('--concept_norm', type=str, default='last')
parser.add_argument('--concept_bias', type=str_to_bool, default=True)
parser.add_argument('--general_stat', type=str_to_bool, default=True)
parser.add_argument('--rnn_dim', type=int, default=128, help='')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# if args.whole_window and args.model == 'jointGRU':
#     args.model = 'maskGRU'
# if not args.whole_window and args.model == 'maskGRU':
#     args.model = 'jointGRU'

import platform
if platform.system() != 'Windows':
    args.num_workers = 0
else:
    torch.cuda.set_per_process_memory_fraction(48/61, 0)

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.enc_in, args.c_out = data_settings[args.dataset][args.features]
args.data_path = data_settings[args.dataset]['data']
args.dec_in = args.enc_in
args.model_id = f'{args.dataset}_{args.seq_len}_{args.pred_len}_{args.model}'

args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'

# args.border_type = 'online'
args.border = settings.get_border(args)

args.wrap_data_class.append(data_loader.Dataset_Concept_Window)

if args.online_method:
    args.patience = 3
    args.save_opt = True
    Exp = getattr(exp_online, 'Exp_' + args.online_method)
    if 'FSNet' in args.model:
        args.pretrain = False
    elif args.online_method.lower() in settings.peft_methods + ['concept_tune', 'drift_tune'] + settings.no_extra_param:
        args.pretrain = True
else:
    Exp = Exp_Concept_Predict

# if not args.general_stat:
#     args.model = 'statGRU'

if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    args.gpu = args.local_rank
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.num_gpus = torch.cuda.device_count()
    args.batch_size = args.batch_size // args.num_gpus

if args.model in settings.need_x_mark:
    args.timeenc = 2
    args.optim = 'AdamW'
    args.patience = 3

args.find_unused_parameters = args.model in ['MTGNN']

data_name = args.data_path.split("/")[-1].split(".")[0]
if platform.system() != 'Windows':
    path = './'
else:
    path = 'D:/data/'
args.concept_path = path + f'concept/{data_name}_{args.seq_len}_192' \
                      f'_span{args.span}_{args.concept_norm}Norm_bias{args.concept_bias}_penalty{args.penalty}.npz'

print('Args in experiment:')
print(args)
# return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark=False
    # torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # args = get_args()
    train_data, train_loader, vali_data, vali_loader = None, None, None, None
    test_data, test_loader = None, None

    if args.is_training:
        all_results = {'mse': [], 'mae': []}
        for ii in range(args.itr):
            if ii == 0 and args.skip:
                continue
            fix_seed = 2023 + ii
            setup_seed(fix_seed)
            print('Seed:', fix_seed)
            # setting record of experiments

            data_name = args.data_path.split("/")[-1].split(".")[0]
            setting = f'{data_name}_sl{args.seq_len}_pl{args.pred_len}_{args.model}_' \
                      f'lr{args.learning_rate}_opt{args.optim}_' \
                      f'dim{args.rnn_dim}_'\
                      f'span{args.span}_{args.concept_norm}Norm_bias{args.concept_bias}_penalty{args.penalty}_{ii}'

            outputs = []
            exp = Exp(args)  # set experiments

            path = os.path.join("checkpoints", setting, 'checkpoint.pth')
            if not args.only_test and not args.do_valid or not os.path.exists(path):
                # if not (args.pretrain and args.online_method in settings.no_extra_param) or args.model in ['FSNet', 'FSNet_Ensemble']:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                _, train_data, train_loader, vali_data, vali_loader = exp.train(setting, train_data, train_loader, vali_data, vali_loader)
            else:
                print('Loading', path)
                exp.load_checkpoint(path)
                print('Learning rate of model_optim is', exp.model_optim.param_groups[0]['lr'])

            if args.online_learning_rate is not None:
                for j in range(len(exp.model_optim.param_groups)):
                    exp.model_optim.param_groups[j]['lr'] = args.online_learning_rate
                print('Adjust learning rate of model_optim to', exp.model_optim.param_groups[0]['lr'])

            if args.do_valid and args.online_method:
                assert isinstance(exp, Exp_Online)
                mse = exp.online(phase='val')[0]
                print('Best Valid MSE:', mse)
                exit()

            if args.do_predict:
                outputs = exp.inference_train(train_data)

            if not args.wo_test and not args.train_only and args.local_rank <= 0:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                if isinstance(exp, Exp_Online):
                    outputs_valid = exp.update_valid(vali_data.dataset if isinstance(vali_data, data_loader.Dataset_Recent)
                                                     else None)
                    if args.do_predict:
                        outputs += outputs_valid
                        del outputs_valid
                        mse, mae, test_data, outputs_test = exp.online(test_data)
                        outputs += outputs_test
                        del outputs_test
                    else:
                        mse, mae, test_data = exp.online(test_data)
                else:
                    mse, mae, test_data, test_loader = exp.test(setting, test_data, test_loader)
                all_results['mse'].append(mse)
            torch.cuda.empty_cache()

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                np.save('./results/' + setting + '.npy', np.concatenate(outputs, 0))

        if not args.wo_test and not args.train_only and args.local_rank <= 0:
            for k in all_results.keys():
                all_results[k] = np.array(all_results[k])
                all_results[k] = [all_results[k].mean(), all_results[k].std()]
            pprint(all_results)
    else:
        ii = 0
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_uni{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.border_type if args.border_type else args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.learning_rate,
            args.univariate,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        args.load_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        # args.pretrain = True
        if args.lift:
            setting += '_lift'

        exp = Exp(args)  # set experiments

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
