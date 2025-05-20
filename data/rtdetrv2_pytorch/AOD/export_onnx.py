import torch
import torch.nn as nn
import argparse
import os
from netbest import PAODNet  # Importa tu red

def main(args):
    """Exporta el modelo a ONNX"""
    # Inicializa el modelo
    model = PAODNet()
    
    # Cargar pesos si se proporciona un archivo .pth
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        print("Advertencia: No se cargaron pesos preentrenados. Usando el estado inicial del modelo.")

    # Pasa el modelo a modo de evaluación
    model.eval()

    # Generar datos de prueba
    data = torch.rand(1, 3, 640, 640)  # Imagen de entrada de prueba
    size = torch.tensor([[640, 640]])  # Tamaño de imagen original

    # Exportar a ONNX
    dynamic_axes = {
        'images': {0: 'N'},  # Batch dinámico
        'orig_target_sizes': {0: 'N'}  # Batch dinámico
    }

    torch.onnx.export(
        model,
        (data,),
        args.output_file,
        input_names=['images'],
        output_names=['clean_image'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    print(f"Modelo exportado exitosamente a {args.output_file}")

    # Validar el modelo ONNX si se especifica
    if args.check:
        import onnx
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print("Validación del modelo ONNX completada.")

    # Simplificar el modelo ONNX si se especifica
    if args.simplify:
        import onnx
        import onnxsim
        dynamic = True
        input_shapes = {'images': data.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(
            args.output_file, input_shapes=input_shapes, dynamic_input_shape=dynamic
        )
        onnx.save(onnx_model_simplify, args.output_file)
        print(f"Modelo ONNX simplificado: {check}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', '-r', type=str, help="Ruta del archivo .pth con los pesos del modelo")
    parser.add_argument('--output_file', '-o', type=str, default="model.onnx", help="Ruta de salida para el modelo ONNX")
    parser.add_argument('--check', action='store_true', default=False, help="Validar el modelo ONNX después de exportarlo")
    parser.add_argument('--simplify', action='store_true', default=False, help="Simplificar el modelo ONNX exportado")

    args = parser.parse_args()

    main(args)
