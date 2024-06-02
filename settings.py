import os
import re

need_x_y_mark = ['Autoformer', 'Transformer', 'Informer']
need_x_mark = ['TCN', 'FSNet', 'OneNet']
need_x_mark += [name + '_Ensemble' for name in need_x_mark]
no_extra_param = ['Online', 'ER', 'DERpp']
peft_methods = ['lora', 'adapter', 'ssf', 'mam_adapter']

data_settings = {
    'wind_N2': {'data': 'wind_N2.csv', 'T':'FR51', 'M':[254, 254], 'prefetch_batch_size': 16},
    'wind': {'data': 'wind.csv', 'T':'UK', 'M':[28,28], 'prefetch_batch_size': 64},
    'ECL':{'data':'electricity.csv','T':'OT','M':[321,321],'S':[1,1],'MS':[321,1], 'prefetch_batch_size': 10},
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'Solar':{'data':'solar_AL.txt','T': 136,'M':[137,137],'S':[1,1],'MS':[137,1], 'prefetch_batch_size': 32},
    'Weather':{'data':'weather.csv','T':'OT','M':[21,21],'S':[1,1],'MS':[21,1], 'prefetch_batch_size': 64},
    'WTH':{'data':'WTH.csv','T':'OT','M':[12,12],'S':[1,1],'MS':[12,1], 'prefetch_batch_size': 64},
    'Traffic': {'data': 'traffic.csv', 'T':'OT', 'M':[862,862], 'prefetch_batch_size': 2},
    'METR_LA': {'data':'metr-la.csv','T': '773869','M':[207,207],'S':[1,1],'MS':[207,1], 'prefetch_batch_size': 16},
    'PEMS_BAY': {'data':'pems-bay.csv','T': 400001,'M':[325,325],'S':[1,1],'MS':[325,1], 'prefetch_batch_size': 10},
    'NYC_BIKE': {'data':'nyc-bike.h5','T': 0,'M':[500,500],'S':[1,1],'MS':[500,1], 'prefetch_batch_size': 4, 'feat_dim': 2},
    'NYC_TAXI': {'data':'nyc-taxi.h5','T': 0,'M':[532,532],'S':[1,1],'MS':[532,1], 'prefetch_batch_size': 4, 'feat_dim': 2},
    'PeMSD4': {'data':'PeMSD4/PeMSD4.npz','T': 0,'M':[921,921],'S':[1,1],'MS':[921,1], 'prefetch_batch_size': 2, 'feat_dim': 3},
    'PeMSD8': {'data':'PeMSD8/PeMSD8.npz','T': 0,'M':[510,510],'S':[1,1],'MS':[510,1], 'prefetch_batch_size': 6, 'feat_dim': 3},
    'Exchange': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'exchange_rate': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'Illness': {'data': 'illness.csv', 'T':'OT', 'M':[7,7], 'prefetch_batch_size': 128},
}

def get_border(args):
    if args.border_type == 'online':
        if args.data.startswith('ETTh'):
            border1s = [0, 4*30*24 - args.seq_len, 5*30*24 - args.seq_len]
            border2s = [4*30*24, 5*30*24, 20*30*24]
            return border1s, border2s
        elif args.data.startswith('ETTm'):
            border1s = [0, 4*30*24*4 - args.seq_len, 5*30*24*4 - args.seq_len]
            border2s = [4*30*24*4, 5*30*24*4, 20*30*24*4]
            return border1s, border2s
        else:
            return 0.2, 0.75
    else:
        return None

hyperparams = {
    'PatchTST': {'e_layers': 3, 'patience': 5},
    'MTGNN': {},
    'LightCTS': {},
    'Crossformer': {'lradj': 'Crossformer', 'e_layers': 3, 'seg_len': 24, 'd_ff': 512, 'd_model': 256, 'n_heads': 4, 'dropout': 0.2},
    'DLinear': {},
    'GPT4TS': {'e_layers': 3, 'd_model': 768, 'n_heads': 4, 'd_ff': 768, 'dropout': 0.3}
}

def get_hyperparams(data, model, args, reduce_bs=True):
    hyperparam: dict = hyperparams[model]

    # if data in 'ECL|PeMSD4|PeMSD8|PEMS_BAY'.split('|'):
    #     hyperparam['temperature'] = 0.1
    # elif data == 'METR_LA':
    #     hyperparam['temperature'] = 2.0
    # else:
    #     hyperparam['temperature'] = 1.0


    if model == 'PatchTST':
        hyperparam['patience'] = max(hyperparam['patience'], args.patience)
        # if data in ['ECL']:
        #     hyperparam['patience'] = 10

        if data in ['ETTh1', 'ETTh2', 'Weather', 'ETTm1', 'ETTm2', 'Exchange']:
            hyperparam['batch_size'] = 128
        elif data in ['Illness']:
            hyperparam['batch_size'] = 16

        if reduce_bs:
            if data in ['PeMSD4']:
                hyperparam['batch_size'] = 4
            elif data in ['Traffic']:
                hyperparam['batch_size'] = 8
            elif data in ['NYC_BIKE', 'NYC_TAXI', 'PeMSD8']:
                hyperparam['batch_size'] = 12
            elif data in ['ECL', 'PEMS_BAY', ]:
                hyperparam['batch_size'] = 16


        if args.lradj != 'type3':
            if data in ['ETTh1', 'ETTh2', 'Weather', 'Exchange', 'wind']:
                hyperparam['lradj'] = 'type3'
            elif data in ['Illness']:
                hyperparam['lradj'] = 'constant'
            else:
                hyperparam['lradj'] = 'TST'
                # if data in ['ETTm1', 'ETTm2']:
                #     hyperparam['pct_start'] = 0.4
                # if data in ['PEMS_BAY', 'METR_LA', 'PeMSD4', 'PeMSD8', 'wind_N2'] and args.pct_start == 0.3:
                #     hyperparam['pct_start'] = 0.2

        if data in ['ETTh1', 'ETTh2', 'Illness']:
            hyperparam.update(**{'dropout': 0.3, 'fc_dropout': 0.3, 'n_heads': 4, 'd_model': 16, 'd_ff': 128})
        elif data in ['ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 128, 'd_ff': 256})
        else:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 64, 'd_ff': 128})

    elif model in ['MTGNN', 'LightCTS']:
        # if data not in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'NYC_BIKE', 'NYC_TAXI', 'Exchange', 'Illness']:
        #     hyperparam['itr'] = 1

        if data in ['ETTh1', 'ETTh2', 'Weather', 'ETTm1', 'ETTm2']:
            hyperparam['batch_size'] = 32

        if reduce_bs:
            if data in ['PeMSD4']:
                hyperparam['batch_size'] = 12
            elif data in ['Solar', 'Exchange']:
                hyperparam['batch_size'] = 32
            elif data in ['Traffic']:
                hyperparam['batch_size'] = 5
            elif data in ['NYC_BIKE', 'NYC_TAXI', 'PeMSD8']:
                hyperparam['batch_size'] = 12
            elif data in ['ECL', 'PEMS_BAY', ]:
                hyperparam['batch_size'] = 16
        else:
            if data in ['Traffic'] and args.pred_len >= 720:
                hyperparam['batch_size'] = 24

        if data in ['Exchange', 'Weather', 'wind']:
            hyperparam['subgraph_size'] = 8
        elif data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Illness']:
            hyperparam['subgraph_size'] = 4

    elif model == 'Crossformer':
        if data == 'ECL' or args.lradj == 'fixed':
            hyperparam['lradj'] = 'fixed'

        # if data not in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Illness']:
        #     hyperparam['itr'] = 1

        if reduce_bs:
            if data in ['PeMSD4']:
                hyperparam['batch_size'] = 4
            elif data in ['Traffic']:
                hyperparam['batch_size'] = 4
            elif data in ['NYC_BIKE', 'NYC_TAXI', 'PeMSD8']:
                hyperparam['batch_size'] = 8
        else:
            if data in ['Traffic', 'PeMSD4'] and args.pred_len >= 720:
                hyperparam['batch_size'] = 24
            if data in ['PeMSD8'] and args.pred_len >= 720:
                hyperparam['batch_size'] = 16

        if data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Illness', 'wind', 'Exchange']:
            hyperparam['d_model'] = 256
            hyperparam['n_heads'] = 4
        else:
            hyperparam['d_model'] = 64
            hyperparam['n_heads'] = 2

        if data in ['Traffic', 'ECL']:
            hyperparam['d_ff'] = 128

        if data in ['Illness']:
            hyperparam['e_layers'] = 2

    elif model == 'GPT4TS':
        if data == 'ETTh1':
            hyperparam['lradj'] = 'typy4'
            hyperparam['tmax'] = 20
            # hyperparam['label_len'] = 168
        elif data == 'ETTh2':
            hyperparam['dropout'] = 1
            hyperparam['tmax'] = 20
            # hyperparam['label_len'] = 168
        elif data == 'Traffic':
            hyperparam['dropout'] = 0.3
        elif data == 'ECL':
            hyperparam['tmax'] = 10
        elif data == 'Illness':
            hyperparam['patch_size'] = 24
            # hyperparam['label_len'] = 18
            hyperparam['batch_size'] = 16

        if data in ['ETTm1', 'ETTm2', 'ECL', 'Traffic', 'Weather', 'WTH']:
            hyperparam['seq_len'] = 512

        if data.startswith('ETTm'):
            hyperparam['stride'] = 16
        elif args.seq_len == 104:
            hyperparam['stride'] = 2

    return hyperparam


def begin_valid_epoch(data, model, H, lr, early_stop, K, tau, state, pct_start=0.2):
    try:
        tmp_lr = str(lr)
        filename = f'logs/LIFT/{model}_LIFT_max_share_{data}_{H}_K{K}_tau{tau}_state{state}_lr{tmp_lr}.log'
        while not os.path.exists(filename) and float(tmp_lr) < 1:
            times = 2 if tmp_lr[-1] == '5' else 5
            tmp_lr = str(float(tmp_lr) * times)
            filename = f'logs/LIFT/{model}_LIFT_max_share_{data}_{H}_K{K}_tau{tau}_state{state}_lr{tmp_lr}.log'

        if not os.path.exists(filename):
            tmp_lr = str(lr)
            filename = f'logs/LIFT/{model}_LIFT_max_share_{data}_{H}_K{K}_state{state}_lr{tmp_lr}.log'
            while not os.path.exists(filename) and float(tmp_lr) < 1:
                times = 2 if tmp_lr[-1] == '5' else 5
                tmp_lr = str(float(tmp_lr) * times)
                filename = f'logs/LIFT/{model}_LIFT_max_share_{data}_{H}_K{K}_state{state}_lr{tmp_lr}.log'

        with open(filename) as f:
            for line in f.readlines()[::-1]:
                s = re.search(r'EarlyStopping counter: (\d+)', line)
                if s is not None:
                    early_stop = int(s.group(1))
                else:
                    s = re.search(r'Epoch: (\d+)', line)
                    if s is not None:
                        epoch = int(s.group(1))
                        return max(epoch - 2 * early_stop, 1)
        return max(expect_min_epoch(data, model, lr) - int(1 + 10 * pct_start) * early_stop, 1)
    except:
        return max(expect_min_epoch(data, model, lr) - int(1 + 10 * pct_start) * early_stop, 1)


def expect_min_epoch(data, model, lr):
    if model == 'PatchTST':
        if data == 'ECL':
            return 20
        if data == 'Solar':
            return 12
        if data in ['ETTm1', 'ETTm2']:
            return 30
        return 5
    return 1


def pretrain_ssf(dataset):
    if dataset in ['ETTh2', 'Weather']:
        return 0.00003
    if dataset == 'ETTm1':
        return 0.0003
    if dataset == 'Traffic':
        return 0.0001
    if dataset == 'ECL':
        return 0.001

def pretrain_lr(model, dataset, H, lr):
    if model == 'MTGNN':
        if dataset in 'Weather|ETTh1|ETTm1'.split('|'):
            return 0.0001
        elif dataset in 'ETTm2'.split('|'):
            return 0.0005
        elif dataset in 'ETTh2'.split('|'):
            return 0.001
        elif dataset in 'Solar'.split('|'):
            return 0.001
        elif dataset in ['ECL']:
            return 0.0005 if H == 720 else 0.001
        return 0.001
    if 'PatchTST' in model:
        if dataset in ['PeMSD8', 'Solar']:
            return 0.001
        return 0.0001
    if model == 'Crossformer':
        if dataset in ['ECL']:
            return 0.005
        if dataset in ['PeMSD8']:
            return 0.001
        elif dataset in ['wind']:
            if H <= 96:
                return 0.0001
            else:
                return 0.00005
        elif dataset in ['Weather']:
            if H >= 192:
                return 0.00001
            else:
                return 0.00005
        elif dataset in 'Solar'.split('|'):
            if H >= 192:
                return 0.0005
            else:
                return 0.001
        elif dataset in 'ETTh1|ETTh2'.split('|'):
            if H >= 168:
                return "0.00001"
            else:
                return 0.0001
        elif dataset in 'ETTm1'.split('|'):
            if H in [192, 336]:
                return "0.00001"
            else:
                return 0.0001
        if dataset in 'ETTm2'.split('|'):
            if H >= 288:
                return "0.00001"
            else:
                return 0.0001
        if dataset in ['Traffic']:
            if H in [720]:
                return 0.0005
            else:
                return 0.001
    return lr


pretrain_lr_online_dict = {
     'TCN': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.001, 'Weather': 0.001, 'Traffic': 0.003},
     'TCN_Ensemble': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.0003, 'Weather': 0.001, 'Traffic': 0.003},
     'FSNet': {'ECL': 0.003, 'ETTh2': 0.001, 'ETTm1': 0.001, 'Weather': 0.001, 'Traffic': 0.003},
    'GPT4TS': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.0001, 'ECL': 0.0001, 'WTH': 0.0003}
}

def get_span_len(args):
    span, prompt_len = 12, 7
    if args.dataset in ['Weather', 'ETTm1']:
        span = 6
    if args.seq_len >= 336:
        if args.dataset in ['ECL']:
            prompt_len = 7
            span = 12
        if args.dataset == 'Weather':
            prompt_len = 12
    else:
        if args.dataset == 'ETTh2':
            prompt_len = 14
        if args.dataset == 'ECL':
            prompt_len = 14
            span = 24
    return span, prompt_len

def get_concept_pred_lr(args):
    data, L, H = args.dataset, args.seq_len, args.pred_len
    if args.concept_norm != 'instance':
        concept_pred_lr_dict = {
            'ECL': 0.001, 'ETTh2': 0.003, 'Weather': 0.003, 'Traffic': 0.003, 'ETTm1': 0.003, 'WTH': 0.001,
        }
        return concept_pred_lr_dict[data]
    if L == 60:
        concept_pred_lr_dict = {
            'ECL': 0.001, 'ETTh2': 0.0001, 'Weather': 0.003, 'Traffic': 0.01, 'ETTm1': 0.003, 'WTH': 0.001,
        }
        if args.dataset == 'ETTh2':
            return 0.003 if H <= 48 else 0.01
        return concept_pred_lr_dict[data]
    else:
        concept_pred_lr_dict = {
            'ECL': 0.001, 'ETTh2': 0.0003, 'Weather': 0.001, 'Traffic': 0.003, 'ETTm1': 0.003, 'WTH': 0.003,
        }
        if args.dataset == 'ETTh2':
            return 0.001 if H <= 24 else 0.0003
        elif args.dataset == 'ETTm1':
            return 0.003 if H <= 24 else 0.001
        elif args.dataset == 'Traffic':
            return 0.001 if H <= 48 else 0.003
        return concept_pred_lr_dict[data]


concept_pred_lr_dict = {
    'ECL': 0.001, 'ETTh2': 0.0003, 'Weather': 0.003, 'Traffic': 0.003
}
concept_pred_warmup_dict = {
    'ECL': 5, 'ETTh2': 10, 'Weather': 10
}