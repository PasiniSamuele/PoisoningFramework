
import os
import sys
import yaml

sys.path = ["./"] + sys.path

import argparse
import numpy as np
from attack.prototype import NormalCase


if __name__ == '__main__':
    normal_train_process = NormalCase()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = normal_train_process.set_args(parser)
    args = parser.parse_args()
    normal_train_process.add_yaml_to_args(args)
    args = normal_train_process.process_args(args)
    normal_train_process.prepare(args)
    normal_train_process.stage1_non_training_data_prepare()
    normal_train_process.stage2_training()