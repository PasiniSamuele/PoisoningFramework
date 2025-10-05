# idea: this file is for the poison sample index selection,
#   generate_single_target_attack_train_poison_index is for all-to-one attack label transform
#   generate_poison_index_from_label_transform aggregate both all-to-one and all-to-all case.

import sys, logging
sys.path.append('../')
import random
import numpy as np
from typing import Callable, Union, List


def generate_single_target_attack_train_poison_index(
        targets:Union[np.ndarray, List],
        tlabel: int,
        pratio: Union[float, None] = None,
        p_num: Union[int,None] = None,
        clean_label: bool = False,
        train : bool = True,
) -> np.ndarray:
    '''
    # idea: given the following information, which samples will be used to poison will be determined automatically.

    :param targets: y array of clean dataset that tend to do poison
    :param tlabel: target label in backdoor attack

    :param pratio: poison ratio, if the whole dataset size = 1
    :param p_num: poison data number, more precise
    need one of pratio and pnum

    :param clean_label: whether use clean label logic to select
    :param train: train or test phase (if test phase the pratio will be close to 1 no matter how you set)
    :return: one-hot array to indicate which of samples is selected
    '''
    targets = np.array(targets)
    logging.debug(f'DEBUG: targets length={len(targets)}, tlabel={tlabel}, pratio={pratio}, p_num={p_num}, clean_label={clean_label}, train={train}')
    logging.debug('Reminder: plz note that if p_num or pratio exceed the number of possible candidate samples\n then only maximum number of samples will be applied')
    logging.debug('Reminder: priority p_num > pratio, and choosing fix number of sample is prefered if possible ')
    poison_index = np.zeros(len(targets))
    if train == False:
        # For test dataset, respect pratio/p_num like training dataset
        non_target_samples = np.where(targets != tlabel)[0]
        if len(non_target_samples) > 0:
            if p_num is not None:
                poison_num = min(p_num, len(non_target_samples))
                logging.debug(f'DEBUG: Test dataset using p_num={p_num}, actual poison_num={poison_num}')
                selected_samples = np.random.choice(non_target_samples, poison_num, replace=False)
                poison_index[selected_samples] = 1
            elif pratio is not None and pratio > 0:
                # For test dataset, apply pratio to non-target samples
                poison_num = max(1, round(pratio * len(non_target_samples)))
                poison_num = min(poison_num, len(non_target_samples))
                logging.debug(f'DEBUG: Test dataset using pratio={pratio}, non_target_samples={len(non_target_samples)}, poison_num={poison_num}')
                selected_samples = np.random.choice(non_target_samples, poison_num, replace=False)
                poison_index[selected_samples] = 1
            else:
                # Fallback to original behavior: poison all non-target samples
                logging.debug(f'DEBUG: Test dataset fallback: poisoning all {len(non_target_samples)} non-target samples')
                poison_index[list(non_target_samples)] = 1
        else:
            logging.debug(f'DEBUG: Test dataset has no non-target samples (all samples are target label {tlabel})')
    else:
        #TRAIN !
        if clean_label == False:
            logging.debug(f'DEBUG: Using clean_label=False path')
            # in train state, all2one non-clean-label case NO NEED TO AVOID target class img
            if p_num is not None:
                logging.debug(f'DEBUG: Using p_num={p_num}')
                non_zero_array = np.random.choice(np.arange(len(targets)), p_num, replace = False)
                poison_index[list(non_zero_array)] = 1
            elif pratio is not None and pratio > 0:
                poison_num = max(1, round(pratio * len(targets)))  # Ensure at least 1 sample is poisoned
                logging.debug(f'DEBUG: Using pratio={pratio}, calculated poison_num={poison_num}')
                non_zero_array = np.random.choice(np.arange(len(targets)), poison_num, replace = False)
                poison_index[list(non_zero_array)] = 1
            else:
                logging.debug(f'DEBUG: No valid p_num or pratio provided')
        else:
            logging.debug(f'DEBUG: Using clean_label=True path')
            if p_num is not None:
                logging.debug(f'DEBUG: Using p_num={p_num}')
                non_zero_array = np.random.choice(np.where(targets == tlabel)[0], p_num, replace = False)
                poison_index[list(non_zero_array)] = 1
            elif pratio is not None and pratio > 0:
                target_samples = np.where(targets == tlabel)[0]
                logging.debug(f'DEBUG: Found {len(target_samples)} target samples for label {tlabel}')
                if len(target_samples) > 0:
                    poison_num = max(1, round(pratio * len(targets)))
                    poison_num = min(poison_num, len(target_samples))  # Don't exceed available target samples
                    logging.debug(f'DEBUG: Using pratio={pratio}, calculated poison_num={poison_num}')
                    non_zero_array = np.random.choice(target_samples, poison_num, replace = False)
                    poison_index[list(non_zero_array)] = 1
                else:
                    logging.debug(f'DEBUG: No target samples found for label {tlabel}')
            else:
                logging.debug(f'DEBUG: No valid p_num or pratio provided')
    logging.info(f'poison num:{sum(poison_index)},real pratio:{sum(poison_index) / len(poison_index)}')
    if sum(poison_index) == 0:
        raise SystemExit('No poison sample generated !')
    return poison_index

from utils.bd_label_transform.backdoor_label_transform import *
from typing import Optional
def generate_poison_index_from_label_transform(
        original_labels: Union[np.ndarray, List],
        label_transform: Callable,
        train: bool = True,
        pratio : Union[float,None] = None,
        p_num: Union[int,None] = None,
        clean_label: bool = False,
) -> Optional[np.ndarray]:
    '''

    # idea: aggregate all-to-one case and all-to-all cases, case being used will be determined by given label transformation automatically.

    !only support label_transform with deterministic output value (one sample one fix target label)!

    :param targets: y array of clean dataset that tend to do poison
    :param tlabel: target label in backdoor attack

    :param pratio: poison ratio, if the whole dataset size = 1
    :param p_num: poison data number, more precise
    need one of pratio and pnum

    :param clean_label: whether use clean label logic to select (only in all2one case can be used !!!)
    :param train: train or test phase (if test phase the pratio will be close to 1 no matter how you set)
    :return: one-hot array to indicate which of samples is selected
    '''
    if clean_label:
        logging.warning("clean_label = True! Note that in our implementation poisoning ratio is ALWAYS defined as number of poisoning samples / number of all samples.")
    if isinstance(label_transform, AllToOne_attack):
        # this is both for allToOne normal case and cleanLabel attack
        return generate_single_target_attack_train_poison_index(
            targets = original_labels,
            tlabel = label_transform.target_label,
            pratio = pratio,
            p_num = p_num,
            clean_label = clean_label,
            train = train,
        )

    elif isinstance(label_transform, AllToAll_shiftLabelAttack):
        if train:
            pass
        else:
            p_num = None
            pratio = 1

        if p_num is not None:
            select_position = np.random.choice(len(original_labels), size = p_num, replace=False)
        elif pratio is not None:
            select_position = np.random.choice(len(original_labels), size=round(len(original_labels) * pratio), replace=False)
        else:
            raise SystemExit('p_num or pratio must be given')
        logging.info(f'poison num:{len(select_position)},real pratio:{len(select_position) / len(original_labels)}')

        poison_index = np.zeros(len(original_labels))
        poison_index[select_position] = 1

        return poison_index
    else:
        logging.debug('Not valid label_transform')



