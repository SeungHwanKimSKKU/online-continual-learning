#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general SeungHwanBash/general_exp3.yml --data config_CVPR/data/cifar10/cifar10_nc.yml --agent config_CVPR/agent/mir/mir_02k_ncm.yml
python -u multiple_run.py --general SeungHwanBash/general_exp3.yml --data config_CVPR/data/cifar10/cifar10_nc.yml --agent config_CVPR/agent/mir/mir_05k_ncm.yml
python -u multiple_run.py --general SeungHwanBash/general_exp3.yml --data config_CVPR/data/cifar10/cifar10_nc.yml --agent config_CVPR/agent/mir/mir_1k_ncm.yml
