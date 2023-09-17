# https://www.youtube.com/watch?v=Sx_HioMUtiY&ab_channel=JoãoReis

import cv2
# pip install opencv-python
import time

# Cores classes
COLORS = [(0, 225, 255), (255, 255, 0), (0, 255, 0), (255, 0 , 0)]

# Carrega classes
class_names = []
with open("coco.names", "r") as arquivo:
    class_names = [cada_nome.strip() for cada_nome in arquivo.readlines() ]
    
# Captura do video
cap = cv2.VideoCapture("walking.mp4")

# Carrega pessos da rede neural  -  net de network
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
# pc melhor     - net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")


# Setar paramentros da rede neural
mode1 = cv2.dnn.DetectionModel(net)
mode1.setInputParams(size=(416, 416), scale=(1/255))

# Lendo frames do video
while True:
    
    # Captura do frame
    _, frame = cap.read()
    
    # Começo da contagem dos MS
    start = time.time()
    
    # DETEÇÂO   -   classe = o que ele e | scores = confinça que ele tem do que e quilo | boxes = onde ta na imagem o que ele achou
    classes, scores, boxes = mode1.detect(frame, 0.1, 0.2)
    
    # Fim da contagem dos MS
    end = time.time()
    
    # Percorrer totas as detecções
    for (classid, score, box) in zip(classes, scores, boxes):
        
        # Gerando uma cor para classe - cor da box
        color = COLORS[int(classid) % len(COLORS)]
        
        # Pegando o nome da classe pelo id e o seu score
        label = f"{class_names[classid]} : {score:.2f}"
        
        # desenhando a box da detecção  - pega a foto, os 4 cantos do retangulo, cor e a expessura
        cv2.rectangle(frame, box, color, 2)
        
        # Escrever o nome da classe em cima da box do objeto
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # Calcular o tempo que levou pra fazer a detecção
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"
    
    # Escrever o fps da imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # Mostrando a imagem
    cv2.imshow("detection", frame)
    
    # Espera resposta
    if cv2.waitKey(1) == 27:
        break

    
# Liberação da camera e destroi todas as janelas
cap.release()
cv2.destroyAllWindows()
