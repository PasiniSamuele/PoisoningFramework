'''
Adversarial Neuron Pruning Purifies Backdoored Deep Models

This file is modified based on the following source:
link : https://github.com/csdongxian/ANP_backdoor.
The defense method is called anp.

@article{wu2021adversarial,
        title={Adversarial neuron pruning purifies backdoored deep models},
        author={Wu, Dongxian and Wang, Yisen},
        journal={Advances in Neural Information Processing Systems},
        volume={34},
        pages={16913--16925},
        year={2021}
        }

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. reconstruct some backbone vgg19 and add some backbone such as densenet161 efficientnet mobilenet
    7. save best model which gets the minimum of asr with acc decreased by no more than 10%
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. anp defense:
        a. train the mask of old model
        b. prune the model depend on the mask
    4. test the result and get ASR, ACC, RC 
'''


import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import json

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense

from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
from collections import OrderedDict
import copy

import utils.defense_utils.anp.anp_model as anp_model

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import BackdoorModelTrainer, Metric_Aggregator, ModelTrainerCLS, ModelTrainerCLS_v2, PureCleanModelTrainer, general_plot_for_epoch
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model, partially_load_state_dict
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2



### anp function
def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.include_noise()
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.include_noise()



def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.exclude_noise()
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


def mask_train(args, model, criterion, mask_opt, noise_opt, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels, *additional_info) in enumerate(data_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.anp_steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if args.anp_eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def test(args, model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels, *additional_info) in enumerate(data_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


def detect_backdoor_from_masks(mask_values_list, args):
    """
    Detect if a model is backdoored based on the distribution of mask values.
    
    Args:
        mask_values_list: List of mask values (floats)
        args: Arguments containing detection parameters
        
    Returns:
        is_backdoor: Boolean indicating if backdoor is detected
        detection_stats: Dictionary with detection statistics
    """
    print("-" * 30)
    print("Analyzing mask values for backdoor detection")
    
    mask_tensor = torch.tensor(mask_values_list)
    
    # Use MAD (Median Absolute Deviation) for outlier detection
    consistency_constant = 1.4826
    median = torch.median(mask_tensor)
    mad = consistency_constant * torch.median(torch.abs(mask_tensor - median))
    min_mad = torch.abs(torch.min(mask_tensor) - median) / mad
    
    # Statistics
    mean_val = torch.mean(mask_tensor)
    std_val = torch.std(mask_tensor)
    min_val = torch.min(mask_tensor)
    max_val = torch.max(mask_tensor)
    
    print(f"Mask Statistics:")
    print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")
    print(f"  Median: {median:.4f}, MAD: {mad:.4f}")
    print(f"  Anomaly index (min_mad): {min_mad:.4f}")
    
    # Detection: two methods available - 'mad' (default) and 'percentile'
    method = getattr(args, 'detection_method', 'mad')
    mad_epsilon = getattr(args, 'mad_epsilon', 1e-6)
    pct_cutoff = getattr(args, 'percentile_cutoff', 5.0)
    low_mask_ratio_threshold = getattr(args, 'low_mask_ratio_threshold', 0.02)

    # protective: use mad_safe to avoid division by near-zero MAD
    mad_safe = mad if mad > mad_epsilon else mad_epsilon

    # Count how many neurons have very low mask values (potential backdoor neurons)
    low_threshold_mad = median - (getattr(args, 'mad_threshold', 2.0) * mad_safe)
    num_low_masks = int(torch.sum(mask_tensor < low_threshold_mad).item())
    total_neurons = len(mask_values_list)
    low_mask_ratio = num_low_masks / float(total_neurons)

    # MAD-based decision: require both an anomalous min AND a non-trivial fraction of very-low masks
    min_mad = float(min_mad)
    mad_threshold = getattr(args, 'mad_threshold', 2.0)
    is_backdoor_mad = (min_mad >= mad_threshold) and (low_mask_ratio >= low_mask_ratio_threshold)

    # Percentile-based decision: look at the fraction below percentile cutoff
    cutoff_value = float(np.percentile(mask_values_list, pct_cutoff))
    num_below_pct = int(np.sum(np.array(mask_values_list) <= cutoff_value))
    pct_low_mask_ratio = num_below_pct / float(total_neurons)
    is_backdoor_pct = pct_low_mask_ratio >= low_mask_ratio_threshold

    # Choose final decision based on requested method, but include both metrics in stats
    if method == 'percentile':
        is_backdoor = is_backdoor_pct
    else:
        is_backdoor = is_backdoor_mad

    print(f"Neurons with mask < {low_threshold_mad:.4f}: {num_low_masks}/{total_neurons} ({low_mask_ratio:.2%})")
    print(f"Percentile({pct_cutoff}) cutoff: {cutoff_value:.4f}, below: {num_below_pct}/{total_neurons} ({pct_low_mask_ratio:.2%})")
    if not is_backdoor:
        print("Not a backdoor model (no anomalous cluster of low mask values by selected method)")
    else:
        print("This is a backdoor model (detected anomalous cluster of low mask values)")
    
    detection_stats = {
        "median": float(median),
        "mad": float(mad),
        "mad_safe": float(mad_safe),
        "min_mad": float(min_mad),
        "mean": float(mean_val),
        "std": float(std_val),
        "min": float(min_val),
        "max": float(max_val),
        "threshold_used": mad_threshold,
        "num_low_masks": num_low_masks,
        "total_neurons": total_neurons,
        "low_mask_ratio": float(low_mask_ratio),
        "percentile_cutoff": float(pct_cutoff),
        "num_below_percentile": num_below_pct,
        "percentile_low_mask_ratio": float(pct_low_mask_ratio),
        "detection_method": method
    }
    
    return is_backdoor, detection_stats


def make_json_serializable(obj):
    """Recursively convert tensors and numpy types to native Python types for JSON serialization."""
    # Torch tensor
    if isinstance(obj, torch.Tensor):
        try:
            if obj.numel() == 1:
                return obj.item()
            return obj.detach().cpu().tolist()
        except Exception:
            return str(obj)
    # Numpy scalar/array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    # Dictionary
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    # List / tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    # Other types - return as-is (JSON will error if not serializable)
    return obj


def get_anp_network(
    model_name: str,
    num_classes: int = 10,
    **kwargs,
):
    
    if model_name == 'preactresnet18':
        from utils.defense_utils.anp.anp_model.preact_anp import PreActResNet18
        net = PreActResNet18(num_classes = num_classes, **kwargs)
    elif model_name == 'vgg19_bn':
        net = anp_model.vgg_anp.vgg19_bn(num_classes = num_classes,  **kwargs)
    elif model_name == 'densenet161':
        net = anp_model.den_anp.densenet161(num_classes= num_classes, **kwargs)
    elif model_name == 'mobilenet_v3_large':
        net = anp_model.mobilenet_anp.mobilenet_v3_large(num_classes= num_classes, **kwargs)
    elif model_name == 'efficientnet_b3':
        net = anp_model.eff_anp.efficientnet_b3(num_classes= num_classes, **kwargs)
    elif model_name == 'resnet18_xai':
        from models.resnet_xai import resnet18
        net = resnet18(num_classes=num_classes, norm_layer=kwargs.get('norm_layer', anp_model.NoisyBatchNorm2d))
    elif model_name == 'resnet18_xai_celeba':
        from models.resnet_xai_celeba import resnet18
        net = resnet18(num_classes=num_classes, norm_layer=kwargs.get('norm_layer', anp_model.NoisyBatchNorm2d))
    elif model_name == 'convnext_tiny':
        # net_from_imagenet = convnext_tiny(pretrained=True) #num_classes = num_classes)
        try :
            net = anp_model.conv_anp.convnext_tiny(num_classes= num_classes, **{k:v for k,v in kwargs.items() if k != "pretrained"})
        except :
            net = anp_model.conv_new_anp.convnext_tiny(num_classes= num_classes, **{k:v for k,v in kwargs.items() if k != "pretrained"})
        # partially_load_state_dict(net, net_from_imagenet.state_dict())
        # net = anp_model.convnext_anp.convnext_tiny(num_classes= num_classes, **kwargs)
    elif model_name == 'vit_b_16':
        try :
            from torchvision.transforms import Resize
            net = anp_model.vit_anp.vit_b_16(
                    pretrained = False,
                    # **{k: v for k, v in kwargs.items() if k != "pretrained"}
                )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                    Resize((224, 224)),
                    net,
                )
        except :
            from torchvision.transforms import Resize
            net = anp_model.vit_new_anp.vit_b_16(
                    pretrained = False,
                    # **{k: v for k, v in kwargs.items() if k != "pretrained"}
                )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                    Resize((224, 224)),
                    net,
                )
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


class anp(defense):
    r"""Adversarial Neuron Pruning for Backdoor Detection (Detection Only)
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the clean test data (NO backdoor data needed)
    3. load the model to analyze
    4. anp detection:
        a. train the mask of model using clean data
        b. analyze mask value distribution for outlier detection
        c. determine if model is backdoored based on mask statistics
        d. save detection results to JSON file and stop
    5. NO mitigation/pruning is performed
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        anp.add_arguments(parser)
        args = parser.parse_args()
        anp_method = anp(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = anp_method.defense(args.result_file)
    
    .. Note::
        @article{wu2021adversarial,
        title={Adversarial neuron pruning purifies backdoored deep models},
        author={Wu, Dongxian and Wang, Yisen},
        journal={Advances in Neural Information Processing Systems},
        volume={34},
        pages={16913--16925},
        year={2021}
        }

    Args:
        baisc args: in the base class
        anp_eps (float): the epsilon for the anp defense in the first step to train the mask
        anp_steps (int): the training steps for the anp defense in the first step to train the mask
        anp_alpha (float): the alpha for the anp defense in the first step to train the mask for the loss
        index (str): the index of the clean data
        ratio (float): the ratio of clean data loader
        print_every (int): print results every few iterations
        nb_iter (int): the number of iterations for training

    Update:
        This version focuses on detection only. No mitigation/pruning is performed.
        Results are saved to detection_result.json in the save_path directory.
        The detection is based on analyzing the distribution of learned mask values.

        
    """ 

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', default = False, type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/anp/config.yaml", help='the path of yaml')

        # set the parameter for the anp detection (mitigation parameters removed)
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--print_every', type=int, help='print results every few iterations')
        parser.add_argument('--nb_iter', type=int, help='the number of iterations for training')

        parser.add_argument('--anp_eps', type=float)
        parser.add_argument('--anp_steps', type=int)
        parser.add_argument('--anp_alpha', type=float)

        parser.add_argument('--index', type=str, help='index of clean data')
        # allow using a separate clean-record (same behavior as nc)
        parser.add_argument('--result_file_clean', type=str, help='the location of a clean record to use instead of the attack record')
        parser.add_argument('--use_clean_file', type=lambda x: str(x) in ['True', 'true', '1'], help='whether to use the clean record specified by --result_file_clean')
        
        # Detection parameters
        parser.add_argument('--detection_method', type=str, default='mad', choices=['mad', 'percentile'], 
                          help='method for outlier detection on mask values')
        parser.add_argument('--mad_threshold', type=float, default=2.0,
                          help='MAD threshold for detecting backdoor (default: 2.0)')
        parser.add_argument('--mad_epsilon', type=float, default=1e-6,
                          help='minimum MAD value to avoid division by tiny MAD')
        parser.add_argument('--percentile_cutoff', type=float, default=5.0,
                          help='percentile (0-100) used by percentile detection method')
        parser.add_argument('--low_mask_ratio_threshold', type=float, default=0.02,
                          help='fraction of neurons below percentile cutoff to flag backdoor')



    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        clean_file = 'record/' + (self.args.result_file_clean if 'result_file_clean' in self.args.__dict__ else '')
        # If use_clean_file is True, prefer saving to the clean record folder, otherwise to the attack record folder
        save_path = 'record/' + self.args.result_file_clean + '/defense/anp/' if (hasattr(self.args, 'use_clean_file') and self.args.use_clean_file) else 'record/' + result_file + '/defense/anp/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        # Load the clean record if requested (mirrors nc behavior)
        if hasattr(self.args, 'use_clean_file') and self.args.use_clean_file and hasattr(self.args, 'result_file_clean') and self.args.result_file_clean is not None:
            try:
                self.result_clean = load_attack_result(clean_file + '/clean_model.pt')
            except Exception:
                # fallback: try to load a generic saved file
                try:
                    self.result_clean = load_attack_result(clean_file + '/attack_result.pt')
                except Exception:
                    logging.warning('Failed to load clean record from %s', clean_file)
                    self.result_clean = None
        # Always load the attack record as well
        try:
            self.result = load_attack_result(attack_file + '/attack_result.pt')
        except Exception:
            # fallback: try to load a generic saved file (e.g., clean_model.pt)
            try:
                self.result = load_attack_result(attack_file + '/clean_model.pt')
            except Exception:
                logging.warning('Failed to load attack record from %s', attack_file)
                self.result = None
        
    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
   
    def set_devices(self):
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )
    
    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        args = self.args


        # a. train the mask using ONLY clean data
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length)
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_clean = self.result['clean_train']
        data_set_clean.wrapped_dataset = data_set_without_tran
        data_set_clean.wrap_img_transform = train_tran
        random_sampler = RandomSampler(data_source=data_set_clean, replacement=True,
                                       num_samples=args.print_every * args.batch_size)
        clean_val_loader = DataLoader(data_set_clean, batch_size=args.batch_size,
                                      shuffle=False, sampler=random_sampler, num_workers=0)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_clean_testset = self.result['clean_test']

        data_clean_testset.wrap_img_transform = test_tran
        clean_test_loader = DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
        if hasattr(self.args, 'use_clean_file') and self.args.use_clean_file and hasattr(self, 'result_clean') and self.result_clean is not None:
            state_dict = self.result_clean
        else:
            state_dict = self.result['model']
        net = get_anp_network(args.model, num_classes=args.num_classes, norm_layer=anp_model.NoisyBatchNorm2d)
        load_state_dict(net, orig_state_dict=state_dict)
        net = net.to(args.device)
        criterion = torch.nn.CrossEntropyLoss().to(args.device)

        parameters = list(net.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

        logging.info('Training neuron masks using clean data only...')
        logging.info('Iter \t lr \t Time \t TrainLoss \t TrainACC \t CleanTestLoss \t CleanTestACC')
        nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
        for i in range(nb_repeat):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = mask_train(args, model=net, criterion=criterion, data_loader=clean_val_loader,
                                            mask_opt=mask_optimizer, noise_opt=noise_optimizer)
            cl_test_loss, cl_test_acc = test(args, model=net, criterion=criterion, data_loader=clean_test_loader)
            end = time.time()
            logging.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                (i + 1) * args.print_every, lr, end - start, train_loss, train_acc,
                cl_test_loss, cl_test_acc))
        
        save_mask_scores(net.state_dict(), os.path.join(args.checkpoint_save, 'mask_values.txt'))
        logging.info(f'Mask values saved to {args.checkpoint_save}mask_values.txt')

        # b. Analyze mask values for backdoor detection
        mask_values = read_data(args.checkpoint_save + 'mask_values.txt')
        
        # Extract just the numerical mask values for analysis
        mask_values_only = [float(x[2]) for x in mask_values]
        
        logging.info(f'Analyzing {len(mask_values_only)} neuron mask values for backdoor detection...')
        
        # Perform outlier detection on mask values
        is_backdoor, detection_stats = detect_backdoor_from_masks(mask_values_only, args)
        
        # Prepare detection result
        detection_result = {
            "is_backdoor": is_backdoor,
            "dataset": args.dataset,
            "model": args.model,
            "num_neurons_analyzed": len(mask_values_only),
            "detection_statistics": detection_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        # Save detection results to JSON
        json_output_path = os.path.join(args.save_path, "detection_result.json")
        with open(json_output_path, "w") as f:
            json.dump(make_json_serializable(detection_result), f, indent=4)
        
        logging.info(f"Detection result saved to {json_output_path}")
        logging.info(f"Is backdoor detected: {is_backdoor}")
        
        # Return detection result (no model pruning/mitigation)
        return detection_result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    anp.add_arguments(parser)
    args = parser.parse_args()
    anp_method = anp(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = anp_method.defense(args.result_file)