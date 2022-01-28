#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general SeungHwanBash/general_exp1.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/er/er_1k_ncm.yml
python -u multiple_run.py --general SeungHwanBash/general_exp1.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/er/er_2k_ncm.yml
python -u multiple_run.py --general SeungHwanBash/general_exp1.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/er/er_5k_ncm.yml
