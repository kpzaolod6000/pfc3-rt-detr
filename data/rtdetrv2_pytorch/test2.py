import cv2
captura = cv2.VideoCapture('gente3.mp4')
while (captura.isOpened()):
  
  ret, imagen = captura.read()
  print("open: ", imagen)
  if ret == True:
    cv2.imshow('video', imagen)
    if cv2.waitKey(30) == ord('s'):
      break
  else: break
captura.release()
cv2.destroyAllWindows()