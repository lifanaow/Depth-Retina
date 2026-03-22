import argparse
import glob
import os
import random
from importlib import import_module
from pprint import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data.distributed

from data.data_mono import DepthDataLoader
from utils.arg_utils import parse_unknown
from utils.config import get_config
from utils.misc import count_parameters

def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def build_model(config):
    try:
        module = import_module(f"models.{config.model}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Model {config.model} not found.") from exc
    try:
        get_version = getattr(module, "get_version")
    except AttributeError as exc:
        raise ValueError(f"Model {config.model} has no get_version function.") from exc
    return get_version(config.version_name).build_from_config(config)


def get_trainer(config):
    if "trainer" not in config or config.trainer in (None, ""):
        raise ValueError(f"Trainer not specified. Config: {config}")
    try:
        return getattr(import_module(f"trainers.{config.trainer}_trainer"), "Trainer")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Trainer {config.trainer}_trainer not found.") from exc


def load_state_dict(model, state_dict):
    state_dict = state_dict.get("model", state_dict)
    do_prefix = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for key, value in state_dict.items():
        if key.startswith("module.") and not do_prefix:
            key = key[7:]
        if not key.startswith("module.") and do_prefix:
            key = f"module.{key}"
        state[key] = value
    model.load_state_dict(state)
    print("Loaded successfully")
    return model


def load_wts(model, checkpoint_path):
    return load_state_dict(model, torch.load(checkpoint_path, map_location="cpu"))


def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):
    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        matches = glob.glob(os.path.join(checkpoint_dir, f"*{config.ckpt_pattern}*{ckpt_type}*"))
        if not matches:
            raise ValueError(f"No matches found for the pattern {config.ckpt_pattern}")
        checkpoint = matches[0]
    else:
        return model
    model = load_wts(model, checkpoint)
    print(f"Loaded weights from {checkpoint}")
    return model


def main_worker(gpu, ngpus_per_node, config):
    try:
        seed = config.seed if "seed" in config and config.seed else 43
        fix_random_seed(seed)
        config.gpu = gpu
        model = build_model(config)
        model = load_ckpt(config, model)
        total_params = f"{round(count_parameters(model) / 1e6, 2)}M"
        config.total_params = total_params
        print(f"Total parameters : {total_params}")
        train_loader = DepthDataLoader(config, "train").data
        test_loader = DepthDataLoader(config, "online_eval").data
        trainer = get_trainer(config)(config, model, train_loader, test_loader, device=config.gpu)
        trainer.train()
    finally:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("forkserver")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="depth-retina")
    parser.add_argument("-d", "--dataset", type=str, default="deyewBM")
    parser.add_argument("--trainer", type=str, default=None)

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)
    overwrite_kwargs["model"] = args.model
    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer

    config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)
    config.shared_dict = mp.Manager().dict() if config.use_shared_dict else None
    config.batch_size = config.bs
    config.mode = "train"
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    try:
        node_str = os.environ["SLURM_JOB_NODELIST"].replace("[", "").replace("]", "")
        nodes = node_str.split(",")
        config.world_size = len(nodes)
        config.rank = int(os.environ["SLURM_PROCID"])
    except KeyError:
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    print("Config:")
    pprint(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, ngpus_per_node, config)
