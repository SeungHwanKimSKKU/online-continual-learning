#!/usr/bin/env bash
cd ../..
source online-cl/bin/activate
python -u multiple_run.py --general SeungHwanBash/general_exp3.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/mir/mir_1k.yml
python -u multiple_run.py --general SeungHwanBash/general_exp3.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/mir/mir_2k.yml
python -u multiple_run.py --general SeungHwanBash/general_exp3.yml --data config_CVPR/data/mini_imagenet/mini_imagenet_nc.yml --agent config_CVPR/agent/mir/mir_5k.yml
