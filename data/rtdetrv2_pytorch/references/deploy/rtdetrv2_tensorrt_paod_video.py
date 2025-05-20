"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import contextlib
import collections
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
import torchvision.transforms as T

import tensorrt as trt
import cv2

# ----- Nuevo: importamos netbest -----
import AOD.netbest as netbest

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("pycuda no está instalado. Asegúrate de instalarlo si se requiere la ruta 'cuda'.")

class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

class TRTInference(object):
    def __init__(self, engine_path, device='cuda:0', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        
        if self.backend == 'cuda':
            self.stream = cuda.Stream()

        self.time_profile = TimeProfiler()

    def init(self, ):
        self.dynamic = False 

    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        '''build bindings
        '''
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            # Si la primera dimensión es -1, se asume "dynamic batch"
            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            if self.backend == 'cuda':
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    data = np.random.randn(*shape).astype(dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 
                else:
                    data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr)
            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        '''torch input
        '''
        # Ajuste de shapes o dtypes en caso dinámico
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)
            
            # Asegurarse de que el dtype del binding y del blob coincidan
            if self.bindings[n].data.dtype != blob[n].dtype:
                # Convertimos el blob al dtype del binding
                blob[n] = blob[n].to(self.bindings[n].data.dtype)
            assert self.bindings[n].data.dtype == blob[n].dtype, f'{n} dtype mismatch'

        # Actualizamos direcciones de memoria
        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def async_run_cuda(self, blob):
        '''numpy input
        '''
        for n in self.input_names:
            cuda.memcpy_htod_async(self.bindings_addr[n], blob[n], self.stream)
        
        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)
        
        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].ptr, self.stream)
            outputs[n] = self.bindings[n].data
        
        self.stream.synchronize()
        
        return outputs
    
    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)
        elif self.backend == 'cuda':
            return self.async_run_cuda(blob)

    def synchronize(self, ):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.backend == 'cuda':
            self.stream.synchronize()
    
    def warmup(self, blob, n):
        for _ in range(n):
            _ = self(blob)

    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self(blob)
        return self.time_profile.total / n

    @staticmethod
    def onnx2tensorrt():
        pass

def draw(images, labels, boxes, scores, thrh = 0.6):
    """
    Dibuja bounding boxes en la(s) imagen(es) con PIL.
    """
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        # Filtramos detecciones por threshold
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            # Muestra la etiqueta y la puntuación
            draw.text((b[0], b[1]), text=f"{lab[j].item()} | {scr[j]:.2f}", fill='blue')

        im.save(f'results_{i}.jpg')  # Guarda la imagen con los BBoxes

# ---- NUEVO: función para aplicar el dehazer ----
def dehaze_image(im_pil, dehaze_net, device='cuda:0'):
    """
    Aplica el modelo netbest (PAODNet) para eliminar la niebla.
    :param im_pil: Imagen PIL de entrada.
    :param dehaze_net: Modelo de dehazing (PAODNet).
    :param device: Dispositivo (cpu/cuda).
    """
    # Convertir PIL a numpy y normalizar [0,1]
    data_hazy = (np.asarray(im_pil) / 255.0).astype(np.float32)
    # [H, W, C] -> torch [C, H, W], y batch dimension
    data_hazy = torch.from_numpy(data_hazy).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        clean_image = dehaze_net(data_hazy)

    # [B, C, H, W] -> [H, W, C]
    clean_image = clean_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    clean_image = (clean_image * 255).astype(np.uint8)

    return Image.fromarray(clean_image)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt-file', type=str, required=True,
                        help="Ruta al engine de TensorRT.")
    parser.add_argument('-f', '--im-file', type=str,
                        help="Imagen de prueba (opcional).")
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help="Dispositivo de inferencia: cpu o cuda:0, etc.")
    parser.add_argument('--dehaze-weights', type=str, default='modelo_dehazing.rtf',
                        help="Ruta al archivo .rtf de pesos del modelo AOD.netbest")

    args = parser.parse_args()

    # 1) Carga del modelo TRT
    m = TRTInference(args.trt_file, device=args.device)

    # 2) Carga del modelo de AOD.netbest (dehazer)
    dehaze_net = netbest.PAODNet().to(args.device)
    dehaze_net.load_state_dict(torch.load(args.dehaze_weights, map_location=args.device))
    dehaze_net.eval()  # Importante poner el modelo en eval()

    # 3) Procesar un video
    video_path = '/home/pytorch/data/rtdetrv2_pytorch/output_heavy_haze.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video:", video_path)
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pueden leer más frames. Finalizando...")
            break

        # Convertimos el frame (BGR -> PIL)
        im_pil = Image.fromarray(frame)
        im_pil = im_pil.convert('RGB')  # Asegurar canal RGB

        # ----- Paso de dehazing -----
        im_pil = dehaze_image(im_pil, dehaze_net, device=args.device)

        # Prepara tensores
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms_ = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms_(im_pil)[None]

        # Empaquetamos un diccionario "blob" con las entradas
        blob = {
            'images': im_data.to(args.device), 
            'orig_target_sizes': orig_size
        }

        # Inferencia con el engine de TensorRT
        output = m(blob)

        # Asumiendo que output tiene las claves 'labels', 'boxes', 'scores'
        if 'labels' in output and 'boxes' in output and 'scores' in output:
            draw([im_pil], output['labels'], output['boxes'], output['scores'])

        # Mostrar en ventana
        cv2.imshow('Frame', np.array(im_pil))

        # Salir al presionar 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 4) (Opcional) Procesar una sola imagen, si se pasa -f/--im-file
    #    (ejemplo de uso adicional)
    if args.im_file:
        im_pil = Image.open(args.im_file).convert('RGB')
        im_pil = dehaze_image(im_pil, dehaze_net, device=args.device)

        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        transforms_ = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms_(im_pil)[None]

        blob = {
            'images': im_data.to(args.device), 
            'orig_target_sizes': orig_size
        }

        output = m(blob)
        if 'labels' in output and 'boxes' in output and 'scores' in output:
            draw([im_pil], output['labels'], output['boxes'], output['scores'])
        cv2.imshow('Imagen', np.array(im_pil))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
