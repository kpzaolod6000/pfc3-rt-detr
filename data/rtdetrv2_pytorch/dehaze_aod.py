import torch
import torchvision
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
import AOD.net as net

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
    input_folder = "/home/pytorch/data/rtdetrv2_pytorch/configs/dataset/dataset/coco/results_coco/light_haze"
    output_folder = "/home/pytorch/data/rtdetrv2_pytorch/configs/dataset/dataset/coco/results_coco/results_light_dehazed"
    model_path = "/home/pytorch/data/rtdetrv2_pytorch/AOD/dehazer.pth"

    main(input_folder, output_folder, model_path)