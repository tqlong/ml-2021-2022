import argparse
from time import time

import numpy as np
from PIL import Image
import torch

from dataset import EasyTransforms
from model import DogCatModel


def get_class_map():
    return {'dog': 1, 'cat': 0}


def get_class_names():
    return ['cat', 'dog']


def get_prediction(model, image: Image.Image, device):
    with torch.no_grad():
        image = EasyTransforms.test(image=np.array(image))["image"]
        image = image.unsqueeze(0).to(device)
        output = model(image)
        _, pred = torch.max(output, 1)
        class_idx = pred.item()
        return {'class_idx': class_idx, 'class_name': get_class_names()[class_idx]}


def load_model(model_path):
    net = DogCatModel.load_from_checkpoint(model_path)
    net.eval()

    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    model = load_model(args.model).to(args.device)
    image = Image.open(args.image)
    start = time()
    prediction = get_prediction(model, image, args.device)
    for _ in range(1000):
        get_prediction(model, image, args.device)
    print(prediction, f"duration {time()-start:.4f} sec(s) per 1000 predictions",)


if __name__ == "__main__":
    main()
