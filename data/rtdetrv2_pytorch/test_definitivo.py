import cv2

# Ruta del video
video_path = 'test1.mp4'

try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV no pudo abrir el archivo o dispositivo de video.")
except Exception as e:
    import traceback
    print("Error capturado:")
    print(traceback.format_exc())


# Leer frames en un bucle
while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Si no quedan más frames, salir del bucle
    if not ret:
        print("No se pueden leer más frames. Finalizando...")
        break

    # Mostrar el frame actual
    cv2.imshow('Frame', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Liberar el objeto VideoCapture y cerrar ventanas
cap.release()
cv2.destroyAllWindows()