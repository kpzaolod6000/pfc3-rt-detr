import torch
import torchvision
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
import AOD.net as net
import argparse


def load_model(model_path):
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load(model_path))
    dehaze_net.eval()  # Poner el modelo en modo de evaluación
    return dehaze_net

def dehaze_image(model, image_path, output_folder):
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)  # Normalización
    
    # Convertir a tensor
    data_hazy = torch.from_numpy(data_hazy).float().permute(2, 0, 1).cuda().unsqueeze(0)

    # Inferencia con el modelo
    with torch.no_grad():
        clean_image = model(data_hazy)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    torchvision.utils.save_image(clean_image, output_path)
    print(f"{image_path} -> {output_path} (done)")

def main(input_folder, output_folder, model_path):
    model = load_model(model_path)
    os.makedirs(output_folder, exist_ok=True)

    test_list = glob.glob(os.path.join(input_folder, '*'))
    for image_path in test_list:
        dehaze_image(model, image_path, output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dehaze images using a pre-trained model.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing hazy images.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where dehazed images will be saved.")
    parser.add_argument("model_path", type=str, help="Path to the pre-trained model (.pth file).")
    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.model_path)