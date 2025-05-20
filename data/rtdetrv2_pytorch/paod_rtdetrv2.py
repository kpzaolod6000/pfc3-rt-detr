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
import time
import os

import AOD.net as net
import AOD.netbest as netbest

def draw(images, labels, boxes, scores, threshold=0.5): #0.6
    for i, image in enumerate(images):
        draw = ImageDraw.Draw(image)

        score = scores[i]
        label = labels[i][score > threshold]
        box = boxes[i][score > threshold]
        filtered_scores = scores[i][score > threshold]

        print(f"Imagen {i}:")
        print(f"Etiquetas filtradas: {label}")
        print(f"Puntuaciones filtradas: {filtered_scores}")

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text(
                (b[0], b[1] - 10),
                text=f"GUN {round(filtered_scores[j].item(), 2)}",
                fill='blue'
            )

def dehaze_image(image_pil, dehaze_net):
    data_hazy = (np.asarray(image_pil) / 255.0)
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
    dehaze_net = netbest.PAODNet().cuda()
    dehaze_net.load_state_dict(
        torch.load(
            '/home/pytorch/data/rtdetrv2_pytorch/pths/snapshots/NYU/PAOD_MS_SSIM_L2_G/dehazer_final.pth'
        )
    )

    # path
    video_path = '/home/pytorch/data/rtdetrv2_pytorch/output_light_haze.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error al abrir el video")
        exit()

    # *** 1) Obtener ancho, alto y fps del video para configurar la salida
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # *** 2) Definir el códec y el nombre/ruta del nuevo video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_dir = '/home/pytorch/data/rtdetrv2_pytorch'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output_light_haze_con_deteccion.mp4')

    # *** 3) Crear el objeto de escritura de video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    accumulated_time = 0  # tiempo acumulado en ms
    fps_calculations = []

    while True:
        start_frame_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("No se pueden leer más frames. Finalizando...")
            break

        # Convertir el frame de OpenCV (BGR) a PIL (RGB) antes de dehaze
        # (si `frame` está en BGR, PIL espera RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(frame_rgb)

        # Deshazear
        im_pil = dehaze_image(im_pil, dehaze_net)

        # Preparar entrada para el modelo
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)

        # Pasar por el modelo
        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Dibujar detecciones sobre la imagen
        draw([im_pil], labels, boxes, scores)

        # Convertir de PIL (RGB) a OpenCV (BGR) para mostrar/escribir
        frame_bgr = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

        # *** 4) Guardar el frame con detecciones en el nuevo video
        out.write(frame_bgr)

        # Si deseas ver en tiempo real:
        cv2.imshow('Frame con deteccion', frame_bgr)

        end_frame_time = time.time()
        frame_time_ms = (end_frame_time - start_frame_time) * 1000
        accumulated_time += frame_time_ms
        frame_count += 1

        # Calcular FPS cada 1 segundo
        if accumulated_time >= 1000:
            fps_current = frame_count
            fps_calculations.append(fps_current)
            print(f"FPS: {fps_current} (frames procesados en 1 segundo)")
            frame_count = 0
            accumulated_time = 0

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    print("\nFPS guardados: ", fps_calculations)
    total_fps = sum(fps_calculations)
    average_fps = total_fps / len(fps_calculations) if fps_calculations else 0
    print(f"\nPromedio total de FPS: {average_fps:.2f}")

    # *** 5) Liberar el objeto VideoCapture y VideoWriter
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
