#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general SeungHwanBash/general_exp2.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/er_1k_ncm.yml
python -u multiple_run.py --general SeungHwanBash/general_exp2.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/er_2k_ncm.yml
python -u multiple_run.py --general SeungHwanBash/general_exp2.yml --data config_CVPR/data/cifar100/cifar100_nc.yml --agent config_CVPR/agent/er/er_5k_ncm.yml
