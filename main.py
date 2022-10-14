#carrega as dependencias
import time

import cv2

#class colors
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

#carrega as classes
class_names = []
with open("coco.names.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#video cap
cap = cv2.VideoCapture(0)

#carrega os pesos da rede neural
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg.txt")

#set parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

#leitura de frames do vídeo
while True:
    #cap de frame
    _, frame = cap.read()

    #inicio da contagem (ms)
    start = time.time()

    #detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.1)

    #fim da contagem(ms)
    end = time.time()

    #percorre todas as detecção
    for (classid, score, box) in zip(classes, scores, boxes):

        #gera cor para a classe
        color = COLORS[int(classid) % len(COLORS)]

        #pega o nome da classe de acordo com ID e score
        label = f"{class_names[classid]} : {score}"

        #desenha a box
        cv2.rectangle(frame, box, color, 2)

        #escreve o nome da classe da detecção
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #tempo que levou para realizar a detecção
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    #escreve o FPS da imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    #mostra a imagem
    cv2.imshow("detections", frame)

    #espera da resposta
    if cv2.waitKey(1) == 27:
        break

#libera camera e fecha as janelas
cap.release()
cv2.destoyAllwindows()
