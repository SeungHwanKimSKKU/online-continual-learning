#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general SeungHwanBash/general_exp4.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/er_1k.yml
python -u multiple_run.py --general SeungHwanBash/general_exp4.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/er_2k.yml
python -u multiple_run.py --general SeungHwanBash/general_exp4.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/er_5k.yml
