import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import net

def dehaze_image(image_path, model_path, output_dir):
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)

    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load(model_path))

    clean_image = dehaze_net(data_hazy)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    torchvision.utils.save_image(clean_image, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dehaze images using a trained model")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the directory containing hazy images")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the directory to save dehazed images")

    args = parser.parse_args()

    input_dir = args.input_dir
    model_path = args.model_path
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of images in the input directory
    test_list = glob.glob(os.path.join(input_dir, "*"))

    for image in test_list:
        dehaze_image(image, model_path, output_dir)
        print(f"{image} processed and saved to {output_dir}")
