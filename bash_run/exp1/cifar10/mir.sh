#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general SeungHwanBash/general_exp1.yml --data config_CVPR/data/cifar10/cifar10_nc.yml --agent config_CVPR/agent/mir/mir_02k.yml
python -u multiple_run.py --general SeungHwanBash/general_exp1.yml --data config_CVPR/data/cifar10/cifar10_nc.yml --agent config_CVPR/agent/mir/mir_05k.yml
python -u multiple_run.py --general SeungHwanBash/general_exp1.yml --data config_CVPR/data/cifar10/cifar10_nc.yml --agent config_CVPR/agent/mir/mir_1k.yml
