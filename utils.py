import shutil
import os
import json
import random
import numpy as np
import torch

def update_pi_from_z(net):
    import copy
    model_dict = net.state_dict()
    save_dict = copy.deepcopy(model_dict)
    to_rename_keys = []
    for key in save_dict:
        if 'subspace' in key:
            to_rename_keys.append(key)
    for key in to_rename_keys:
        # print(f'renamed key {key}')
        pre, post = key.split('subspace')
        save_dict[pre + 'cluster' + post] = save_dict.pop(key)

    model_dict.update(save_dict)
    log = net.load_state_dict(model_dict)
    # print(log)
    return net

def init_pipeline(model_dir, args):
    """Initialize folders and Seed for experiments"""
    # project folder
    if os.path.exists(model_dir):
        print('EXP PATH EXISTS, PLEASE BE CAUTIOUS')
    os.makedirs(model_dir,exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'),exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'tensorboard'),exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'codes'),exist_ok=True)

    # save exp settings
    save_params(model_dir, vars(args))

    # GPU and seed setup
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # tensorboard settings
    from torch.utils.tensorboard import SummaryWriter 
    writer = SummaryWriter(os.path.join(model_dir, 'tensorboard'))
    
    # copy codes
    for filepath in os.listdir('./'):
        if filepath.endswith('.py'):
            shutil.copyfile(os.path.join('./',filepath), os.path.join(model_dir,'codes',filepath))
    return writer

def save_params(model_dir, params):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)