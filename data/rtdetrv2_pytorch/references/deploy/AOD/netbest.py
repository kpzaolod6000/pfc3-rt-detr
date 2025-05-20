import torch
import torch.nn as nn
import torch.nn.functional as F

class PfAAM(nn.Module):
    """
    Parameter-free Average Attention Module (PfAAM).
    
    Siguiendo la idea del paper:
      - A_sp(x) promedia a lo largo de canales (dim=1) → [B, 1, H, W]
      - A_ch(x) promedia a lo largo de la dimensión espacial (dim=(2,3)) → [B, C, 1, 1]
      - Se re-expanden ambos para obtener la misma forma que 'x'
      - Se multiplican y pasa por sigmoide, luego se multiplica por 'x'
      
    Ecuación en el paper:
        A_sp(x) = (1/3)*∑(x)  sobre canales
        A_ch(x) = (1/(H·W))*∑(x)  sobre H,W
        F' = σ(A_sp ⊗ A_ch) ⊗ F
    """
    def __init__(self):
        super(PfAAM, self).__init__()

    def forward(self, F):
        # A_sp: promedio a lo largo del canal => [B, 1, H, W]
        A_sp = torch.mean(F, dim=1, keepdim=True)  # 1/3 * sum(x, canal)

        # A_ch: promedio a lo largo de la dimensión espacial => [B, C, 1, 1]
        A_ch = torch.mean(F, dim=(2, 3), keepdim=True)  # 1/(H·W) * sum(x, espacial)

        # Expandir y combinar con multiplicación elemento a elemento
        attention_map = torch.sigmoid(A_sp.expand_as(F) * A_ch.expand_as(F))
        
        # Atender sobre F (element-wise)
        F_prime = attention_map * F
        return F_prime


class PAODNet(nn.Module):
    """
    Ejemplo de uso de PfAAM en una red con varias convoluciones concatenadas.
    
    - Bloques de convolución con distintos kernel_size (1x1, 3x3, 5x5, 7x7, 3x3)
    - Cada salida pasa por PfAAM
    - Al final, la salida limpia se calcula como: f(x) = K(x)*x - K(x) + 1
    """
    def __init__(self):
        super(PAODNet, self).__init__()

        # Definimos convoluciones según el diagrama
        # Entrada y salida de 3 canales en cada una (para imágenes RGB)
        self.e_conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.e_conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.e_conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.e_conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.e_conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.pfaam = PfAAM()

    def forward(self, x):
        # Bloque 1
        x1 = self.pfaam(self.relu(self.e_conv1(x)))
        x2 = self.pfaam(self.relu(self.e_conv2(x1)))

        # Concat1 → Conv3 → PfAAM
        concat1 = torch.cat((x1, x2), dim=1)  # [B, 6, H, W]
        x3 = self.pfaam(self.relu(self.e_conv3(concat1)))

        # Concat2 → Conv4 → PfAAM
        concat2 = torch.cat((x2, x3), dim=1)  # [B, 6, H, W]
        x4 = self.pfaam(self.relu(self.e_conv4(concat2)))

        # Concat3 → Conv5 → PfAAM
        concat3 = torch.cat((x1, x2, x3, x4), dim=1)  # [B, 12, H, W]
        x5 = self.pfaam(self.relu(self.e_conv5(concat3)))

        # Módulo final para generar la imagen limpia:
        # f(x) = K(x)*x - K(x) + b, donde b = 1
        clean_image = self.relu((x5 * x) - x5 + 1)

        return clean_image


# Ejemplo de uso / prueba
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PAODNet().to(device)

    # Imagen de entrada de prueba (batch=1, 3 canales, 256x256)
    input_image = torch.randn(1, 3, 256, 256).to(device)
    output = model(input_image)

    print("Forma de la salida:", output.shape)  # (1, 3, 256, 256)
