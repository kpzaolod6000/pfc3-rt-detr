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

import AOD.net as net

def draw(images, labels, boxes, scores, threshold=0.5): #0.6
    for i, image in enumerate(images):
        draw = ImageDraw.Draw(image)

        # Filtrar por umbral
        score = scores[i]
        label = labels[i][score > threshold]
        box = boxes[i][score > threshold]
        filtered_scores = scores[i][score > threshold]

        # Imprimir etiquetas para depuración
        print(f"Imagen {i}:")
        print(f"Etiquetas filtradas: {label}")
        print(f"Puntuaciones filtradas: {filtered_scores}")

        # Dibujar rectángulos y etiquetas
        for j, b in enumerate(box):
            print(f"Etiquetas filtradas: {label[j].item()}")
            draw.rectangle(list(b), outline='red')
            # draw.text((b[0], b[1]), text=f"{label[j].item()} {round(filtered_scores[j].item(), 2)}", fill='blue')
            draw.text((b[0], b[1]-10), text=f"GUN {round(filtered_scores[j].item(), 2)}", fill='blue')

        # Guardar la imagen anotada
        image.save(f'results_{i}.jpg')


def dehaze_image(image_path, dehaze_net):

    data_hazy = (np.asarray(image_path) / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)

    clean_image = dehaze_net(data_hazy)
    clean_image = clean_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    clean_image = (clean_image * 255).astype(np.uint8)
    return Image.fromarray(clean_image)


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

    ## AOD WEIGHTS
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load('AOD/dehazer.pth'))

    # Ruta del video
    video_path = 'output_light_haze.mp4'

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
    
         # Convertir el frame de OpenCV (BGR) a PIL (RGB)
        im_pil = Image.fromarray(frame)

        # Convertir a RGB explícitamente si no estás seguro del formato
        # im_pil = im_pil.convert('RGB')
        
        im_pil = dehaze_image(im_pil, dehaze_net)
    
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
