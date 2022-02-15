import argparse
import random
import numpy as np
import torch
from experiment.run import multiple_run
from utils.io import load_yaml
from utils.utils import boolean_string
from types import SimpleNamespace
from utils.setup_elements import default_trick
from copy import deepcopy
def main(args):
    genereal_params = load_yaml(args.general)
    data_params = load_yaml(args.data)
    agent_params = load_yaml(args.agent)
    genereal_params['verbose'] = args.verbose
    genereal_params['cuda'] = torch.cuda.is_available()
    genereal_params['sotre'] = args.store
    default_trick_copy = deepcopy(default_trick)
    for i in agent_params:
        if i in default_trick_copy:
            default_trick_copy[i] = True
    genereal_params['trick'] = default_trick_copy
    genereal_params['head']=args.head
    final_params = SimpleNamespace(**genereal_params, **data_params, **agent_params)
    print(final_params)
    # set up seed
    np.random.seed(final_params.seed)
    random.seed(final_params.seed)
    torch.manual_seed(final_params.seed)
    if args.cuda:
        torch.cuda.manual_seed(final_params.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    multiple_run(final_params, store=True)
if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Run with config")
    parser.add_argument('--general', dest='general', default='config/general.yml')
    parser.add_argument('--data', dest='data', default='config/data/cifar100/cifar100_nc.yml')
    parser.add_argument('--agent', dest='agent', default='config/agent/er.yml')
    parser.add_argument('--verbose', type=boolean_string, default=True,
                        help='print information or not (default: %(default)s)')
    parser.add_argument('--store', type=boolean_string, default=True,
                        help='Store result or not (default: %(default)s)')
    parser.add_argument('--save-path', dest='save_path', default=None)
    parser.add_argument('--head',dest='head',type=str,default='mlp')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    main(args)
