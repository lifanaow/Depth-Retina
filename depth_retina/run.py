import argparse
from pprint import pprint

import torch
from utils.easydict import EasyDict as edict
from tqdm import tqdm

from data.data_mono import DepthDataLoader
from models.builder import build_model
from utils.arg_utils import parse_unknown
from utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)
import matplotlib.pyplot as plt
import cv2
import numpy as np


@torch.no_grad()
def infer(model, image, **kwargs):
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    if isinstance(image, torch.Tensor):
        img = image
    else:
        img = np.asarray(image, dtype=np.float32) / 255.0
        img = torch.from_numpy(np.array([img.transpose((2, 0, 1))]))

    pred = model(img, **kwargs)
    pred = get_depth_from_prediction(pred)

    return pred


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        pred = infer(model, image, dataset=sample['dataset'][0], focal=focal)

        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            d = colorize(depth.squeeze().cpu().numpy(), 0, 10)
            p = colorize(pred.squeeze().cpu().numpy(), 0, 10)
            im = transforms.ToPILImage()(image.squeeze().cpu())
            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png"))

        metrics.update(compute_metrics(depth, pred, config=config))

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

def main(config, image):
    model = build_model(config)
    image = np.asarray(image, dtype=np.float32) / 255.0
    img = torch.from_numpy(np.array([image.transpose((2, 0, 1))]))
    pred = infer(model, img)

    return pred

def infer_model(model_name, pretrained_resource, image, **kwargs):

    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "infer", **overwrite)
 
    pprint(config)
    print(f"Inferencing {model_name}:{pretrained_resource}")
    metrics = main(config, image)
    return metrics

def load_model(model_name, pretrained_resource, **kwargs):

    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "infer", **overwrite)

    pprint(config)
    print(f"Inferencing {model_name}:{pretrained_resource}")

    model = build_model(config)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="")
    parser.add_argument("-i", "--image", type=str,
                        required=False, default="", help="")
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    infer_model(args.model, pretrained_resource=args.pretrained_resource, image=args.image, **overwrite_kwargs)
