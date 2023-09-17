from flask import Flask, request, render_template, redirect, url_for
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('fotoCel.html')

@app.route('/upload', methods=['POST'])
def upload():
   imagem = request.files['imagem']
   if imagem:
       # Defina um nome fixo para todas as imagens
       nome_arquivo = 'bicycle_image.jpg'
       caminho_arquivo = os.path.join('uploads', nome_arquivo)
       imagem.save(caminho_arquivo)

       # Chame o script Python main-image.py para processar a imagem
       subprocess.run(['python', 'python-reconhecimento/main-image.py', caminho_arquivo])

       return 'Upload bem-sucedido! A imagem foi processada.'
   return 'Erro no upload da imagem.'

if __name__ == '__main__':
   app.run(debug=True)





import cv2

# Carregar a rede YOLO com os pesos pré-treinados e a configuração
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Carregar as classes
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Definir a camada de saída
output_layer_names = net.getUnconnectedOutLayersNames()

# Carregar a imagem
image = cv2.imread('/upload/bicycle_image.jpg')
height, width, _ = image.shape

# Pré-processamento da imagem
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layer_names)

# Informações de detecção
class_ids = []
confidences = []
boxes = []

# Processar as saídas da rede
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]
        if confidence > 0.5:  # Limiar de confiança
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Aplicar supressão não máxima para eliminar detecções redundantes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Desenhar caixas delimitadoras nas bicicletas detectadas
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Cor da caixa delimitadora (verde)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), font, 1, color, 2)

# Mostrar imagem com detecções
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
