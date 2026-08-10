# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
import sys
from typing import Any
REPO_DIR = "/data/project/users/bassant/code/DINO_LIN/dinov3-echo/"
sys.path.append(REPO_DIR)
import logging
from omegaconf import OmegaConf
import os
import sys
from typing import Any
import wandb

from dinov3.eval.segmentation.config import SegmentationConfig
from dinov3.eval.segmentation.eval import test_segmentation
from dinov3.eval.segmentation.train import train_segmentation
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.setup import load_model_and_context
from dinov3.run.init import job_context


logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-semantic-segmentation.csv"
MAIN_METRICS = ["mIoU"]


def run_segmentation_with_dinov3(
    backbone,
    config,
):
    # wandb.init(project="dinov3", name=f"measurments_Weighted_Dice_segmentation_eval_linear_prob_13_labels_{config.wandb_name}_Upsample_RV")
    # if config.load_from:
    logger.info("Testing model performance on a pretrained decoder head")
    return test_segmentation(backbone=backbone, config=config)
    # assert config.decoder_head.type == "linear", "Only linear head is supported for training"
    # return train_segmentation(backbone=backbone, config=config)


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    if "config" in eval_args:  # using a config yaml file, useful for training
        base_config_path = eval_args.pop("config")
        output_dir = eval_args["output_dir"]
        base_config = OmegaConf.load(base_config_path)
        structured_config = OmegaConf.structured(SegmentationConfig)
        dataclass_config: SegmentationConfig = OmegaConf.to_object(
            OmegaConf.merge(
                structured_config,
                base_config,
                OmegaConf.create(eval_args),
            )
        )
    else:  # either using default values, or only adding some args to the command line
        dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=SegmentationConfig)
    backbone = None
    if dataclass_config.model:
        backbone, _ = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    else:
        assert dataclass_config.load_from == "dinov3_vit7b16_ms"
    logger.info(f"Segmentation Config:\n{OmegaConf.to_yaml(dataclass_config)}")
    segmentation_file_path = os.path.join(output_dir, "segmentation_config.yaml")
    OmegaConf.save(config=dataclass_config, f=segmentation_file_path)
    results_dict = run_segmentation_with_dinov3(backbone=backbone, config=dataclass_config)
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    with job_context(output_dir=eval_args["output_dir"]):
        benchmark_launcher(eval_args=eval_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# if __name__ == "__main__":
#     import os
#     import torch
#     import torch.distributed as dist

#     def setup_distributed_if_needed():
#         """
#         Returns (use_ddp: bool, local_rank: int).
#         If use_ddp is False, we will run single-process single-GPU on device 0 (if CUDA available).
#         """
#         # If torchrun / launcher set LOCAL_RANK, use it. Otherwise fall back to single-process.
#         local_rank_env = os.environ.get("LOCAL_RANK")
#         world_size_env = os.environ.get("WORLD_SIZE")

#         # If user explicitly set WORLD_SIZE=1 earlier, prefer single-process mode
#         if world_size_env is not None and int(world_size_env) == 1:
#             return False, 0

#         if local_rank_env is None:
#             # Not launched via torchrun; check if user wants single-process default
#             if world_size_env is None or int(world_size_env) == 1:
#                 return False, 0
#             # If WORLD_SIZE>1 but LOCAL_RANK not set, try to infer from RANK if on single node
#             if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
#                 # best-effort fallback: compute local from rank % gpus-per-node (not recommended)
#                 try:
#                     rank = int(os.environ["RANK"])
#                     gpus = torch.cuda.device_count() or 1
#                     return True, rank % gpus
#                 except Exception:
#                     return False, 0

#         # Normal multi-process (torchrun / launcher) case:
#         local_rank = int(local_rank_env)
#         # initialize process group (idempotent if already initialized)
#         if not dist.is_initialized():
#             # Make sure MASTER_ADDR/PORT are set in your environment or your launcher did it.
#             dist.init_process_group(backend="nccl", init_method="env://")
#         # set the CUDA device for this process BEFORE creating the model.
#         torch.cuda.set_device(local_rank)
#         return True, local_rank

#     _, local_rank = setup_distributed_if_needed()
#     sys.exit(main())

# # Copyright (c) Meta Platforms, Inc. and affiliates.
# #
# # This software may be used and distributed in accordance with
# # the terms of the DINOv3 License Agreement.

# import logging
# from omegaconf import OmegaConf
# import os
# import sys
# from typing import Any
# REPO_DIR = "/data/project/users/bassant/code/DINOV3/dinov3"
# sys.path.append(REPO_DIR)
# from dinov3.eval.segmentation.config import SegmentationConfig
# from dinov3.eval.segmentation.eval import test_segmentation
# from dinov3.eval.segmentation.train import train_segmentation
# from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
# from dinov3.eval.setup import load_model_and_context
# from dinov3.run.init import job_context
# from dinov3.eval.utils import ModelWithNormalize, average_metrics, evaluate

# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))

# logger = logging.getLogger("dinov3")

# RESULTS_FILENAME = "results-semantic-segmentation.csv"
# MAIN_METRICS = ["mIoU"]


# def run_segmentation_with_dinov3(
#     backbone,
#     config,
#     autocast_dtype,
#     is_train
# ):
#     if not is_train:
#     # if config.load_from:
#         logger.info("Testing model performance on a pretrained decoder head")
#         return test_segmentation(backbone=backbone, config=config)
#         # assert config.decoder_head.type == "linear", "Only linear head is supported for training"
#         # backbone = ModelWithNormalize(backbone)
#     else:
#         return train_segmentation(backbone=backbone, config=config, autocast_dtype = autocast_dtype)


# def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
#     """Initialization of distributed and logging are preconditions for this method"""
#     if "config" in eval_args:  # using a config yaml file, useful for training
#         base_config_path = eval_args.pop("config")
#         output_dir = eval_args["output_dir"]
#         base_config = OmegaConf.load(base_config_path)
#         structured_config = OmegaConf.structured(SegmentationConfig)
#         dataclass_config: SegmentationConfig = OmegaConf.to_object(
#             OmegaConf.merge(
#                 structured_config,
#                 base_config,
#                 OmegaConf.create(eval_args),
#             )
#         )
#     else:  # either using default values, or only adding some args to the command line
#         dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=SegmentationConfig)
#     # backbone = None
#     # if dataclass_config.model:
#     backbone, model_context = load_model_and_context(dataclass_config.model, output_dir=output_dir)
#     # else:
#     #     assert dataclass_config.load_from == "dinov3_vit7b16_ms"
#     logger.info(f"Segmentation Config:\n{OmegaConf.to_yaml(dataclass_config)}")
#     segmentation_file_path = os.path.join(output_dir, "segmentation_config.yaml")
#     OmegaConf.save(config=dataclass_config, f=segmentation_file_path)
#     results_dict = run_segmentation_with_dinov3(backbone=backbone, config=dataclass_config, autocast_dtype=model_context["autocast_dtype"], is_train=False)
#     # write_results(results_dict, output_dir, RESULTS_FILENAME)
#     return results_dict


# def main(argv=None):
#     if argv is None:
#         argv = sys.argv[1:]
#     eval_args = cli_parser(argv)
#     with job_context(output_dir=eval_args["output_dir"]):
#         benchmark_launcher(eval_args=eval_args)
#     return 0


# if __name__ == "__main__":
#     import os
#     import torch
#     import torch.distributed as dist

#     def setup_distributed_if_needed():
#         """
#         Returns (use_ddp: bool, local_rank: int).
#         If use_ddp is False, we will run single-process single-GPU on device 0 (if CUDA available).
#         """
#         # If torchrun / launcher set LOCAL_RANK, use it. Otherwise fall back to single-process.
#         local_rank_env = os.environ.get("LOCAL_RANK")
#         world_size_env = os.environ.get("WORLD_SIZE")

#         # If user explicitly set WORLD_SIZE=1 earlier, prefer single-process mode
#         if world_size_env is not None and int(world_size_env) == 1:
#             return False, 0

#         if local_rank_env is None:
#             # Not launched via torchrun; check if user wants single-process default
#             if world_size_env is None or int(world_size_env) == 1:
#                 return False, 0
#             # If WORLD_SIZE>1 but LOCAL_RANK not set, try to infer from RANK if on single node
#             if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
#                 # best-effort fallback: compute local from rank % gpus-per-node (not recommended)
#                 try:
#                     rank = int(os.environ["RANK"])
#                     gpus = torch.cuda.device_count() or 1
#                     return True, rank % gpus
#                 except Exception:
#                     return False, 0

#         # Normal multi-process (torchrun / launcher) case:
#         local_rank = int(local_rank_env)
#         # initialize process group (idempotent if already initialized)
#         if not dist.is_initialized():
#             # Make sure MASTER_ADDR/PORT are set in your environment or your launcher did it.
#             dist.init_process_group(backend="nccl", init_method="env://")
#         # set the CUDA device for this process BEFORE creating the model.
#         torch.cuda.set_device(local_rank)
#         return True, local_rank

#     _, local_rank = setup_distributed_if_needed()
#     sys.exit(main())
