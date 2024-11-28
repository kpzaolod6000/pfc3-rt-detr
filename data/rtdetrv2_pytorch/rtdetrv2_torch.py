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


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    # Ruta del video
    video_path = 'video.mp4'

    # Crear un objeto VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir el video")
        exit()

    # Leer frames en un bucle
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        # print("frame.size:", frame.shape[:2])
        # print("type of frame.size:", type(frame))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # w, h = frame.shape[:2]
         # Convertir el frame de OpenCV (BGR) a PIL (RGB)
        im_pil = Image.fromarray(frame)

        # Convertir a RGB explícitamente si no estás seguro del formato
        im_pil = im_pil.convert('RGB')
    
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        draw([im_pil], labels, boxes, scores)
        
        # Si no quedan más frames, salir del bucle
        if not ret:
            print("No se pueden leer más frames. Finalizando...")
            break

        # Mostrar el frame actual
        frame_bgr = cv2.cvtColor(np.array(im_pil), cv2.COLOR_BGR2RGB)

        # Mostrar el frame procesado
        cv2.imshow('Frame', np.array(im_pil))
        # Salir con la tecla 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Liberar el objeto VideoCapture y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

    # im_pil = Image.open(args.im_file).convert('RGB')
    # w, h = im_pil.size
    # orig_size = torch.tensor([w, h])[None].to(args.device)

    # transforms = T.Compose([
    #     T.Resize((640, 640)),
    #     T.ToTensor(),
    # ])
    # im_data = transforms(im_pil)[None].to(args.device)

    # output = model(im_data, orig_size)
    # labels, boxes, scores = output

    # draw([im_pil], labels, boxes, scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
