"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

from src.core import YAMLConfig
import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def draw(images, labels, boxes, scores, thrh=0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue')

        im.save(f'results_{i}.jpg')


def main(args):
    """main
    """
    # Cargar configuración y modelo
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()

    # Asumiendo que tenemos un archivo de anotaciones COCO
    annotation_file = '/home/pytorch/data/rtdetrv2_pytorch/val2017/annotations/instances_val2017.json'  # Ajustar a tu ruta real
    coco = COCO(annotation_file)

    # Cargar imagen
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    # Inferencia
    with torch.no_grad():
        output = model(im_data, orig_size)

    labels, boxes, scores = output

    # Dibuja resultados
    draw([im_pil.copy()], labels, boxes, scores)

    # Supongamos que el image_id es conocido de antemano (por ejemplo 281179)
    # Debes asegurarte que args.im_file corresponda a la misma imagen del dataset GT.
    image_id = 281179  # Ajustar según tu JSON GT

    # Preparar predicciones en formato COCO
    # boxes en formato [x_min, y_min, x_max, y_max]. Debemos convertir a [x,y,w,h].
    coco_results = []

    # Asumimos un batch de 1 (una sola imagen) y por lo tanto output[0,...] es para esa imagen
    _labels = labels[0].cpu().numpy()
    _boxes = boxes[0].cpu().numpy()
    _scores = scores[0].cpu().numpy()

    for lbl, box, scr in zip(_labels, _boxes, _scores):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        coco_results.append({
            "image_id": image_id,
            "category_id": int(lbl),  # asumiendo que lbl es category_id
            "bbox": [float(x_min), float(y_min), float(width), float(height)],
            "score": float(scr)
        })

    # Cargar predicciones al formato COCO y evaluar
    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-r', '--resume', type=str)
    parser.add_argument('-f', '--im-file', type=str, help='Ruta de la imagen a procesar')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
