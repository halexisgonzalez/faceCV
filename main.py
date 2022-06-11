import cv2
import mediapipe as mp
import math
#import RPi.GPIO as GPIO

#GPIO.setmode(GPIO.BCM)
#GPIO.setup(18, GPIO.OUT)
#GPIO.setup(13, GPIO.OUT)

#pwm1 = GPIO.PWM(18, 50)
#pwm2 = GPIO.PWM(13, 50)
#pwm1.start(0)
#pwm2.start(0)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpDibujo = mp.solutions.drawing_utils
configdibujo = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

mpmallafacial = mp.solutions.face_mesh
mallafacial = mpmallafacial.FaceMesh(max_num_faces=1)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1200, 900))
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = mallafacial.process(frameRGB)
    calibx = []
    caliby = []
    calib = []
    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            for id, puntos in enumerate(rostros.landmark):
                alto, ancho, capas = frame.shape
                x, y = int(puntos.x * ancho), int(puntos.y * alto)
                calibx.append(x)
                caliby.append(y)
                calib.append([id, x, y])
                if len(calib) == 468:
                    # cejaderecha
                    cx1, cy1 = calib[65][1:]
                    cx2, cy2 = calib[158][1:]
                    longitud1calib = math.hypot(cx2 - cx1, cy2 - cy1)
                    # print(longitud1)

                    # cejaizquierda
                    cx3, cy3 = calib[295][1:]
                    cx4, cy4 = calib[385][1:]
                    longitud2calib = math.hypot(cx4 - cx3, cy4 - cy3)
                    # print(longitud2)

                    # comisurasboca
                    cx5, cy5 = calib[78][1:]
                    cx6, cy6 = calib[308][1:]
                    longitud3calib = math.hypot(cx6 - cx5, cy6 - cy5)
                    # print(longitud3)

                    # aperturaboca
                    cx7, cy7 = calib[13][1:]
                    cx8, cy8 = calib[14][1:]
                    longitud4calib = math.hypot(cx8 - cx7, cy8 - cy7)
                    # print(longitud4)

                    # muecaizquierda
                    cx9, cy9 = calib[291][1:]
                    cx10, cy10 = calib[401][1:]
                    longitud5calib = math.hypot(cx10 - cx9, cy10 - cy9)
                    # print(longitud5)

                    # muecaderecha
                    cx11, cy11 = calib[61][1:]
                    cx12, cy12 = calib[177][1:]
                    longitud6calib = math.hypot(cx12 - cx11, cy12 - cy11)
                    # print(longitud6)
    cv2.imshow('calibracion', cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == 32:  # es la tecla espaciadora
        break
cv2.destroyWindow("calibracion")
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1200, 900))
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = mallafacial.process(frameRGB)
    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame, rostros, mpmallafacial.FACEMESH_CONTOURS, configdibujo, configdibujo)

            for id, puntos in enumerate(rostros.landmark):
                alto, ancho, capas = frame.shape
                x, y = int(puntos.x * ancho), int(puntos.y * alto)

                lista.append([id, x, y])
                if len(lista) == 468:
                    # cejaderecha
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), t)
                    cv2.circle(frame, (x1, y1), r, (0, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), r, (0, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (cx, cy), r, (0, 0, 0), cv2.FILLED)
                    longitud1 = math.hypot(x2 - x1, y2 - y1)
                    # print(longitud1)

                    # cejaizquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    cv2.line(frame, (x3, y3), (x4, y4), (0, 0, 0), t)
                    cv2.circle(frame, (x3, y3), r, (0, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (x4, y4), r, (0, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (cx2, cy2), r, (0, 0, 0), cv2.FILLED)
                    longitud2 = math.hypot(x4 - x3, y4 - y3)
                    # print(longitud2)

                    # comisurasboca
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    cv2.line(frame, (x5, y5), (x6, y6), (0, 0, 0), t)
                    cv2.circle(frame, (x5, y5), r, (0, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (x6, y6), r, (0, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (cx3, cy3), r, (0, 0, 0), cv2.FILLED)
                    longitud3 = math.hypot(x6 - x5, y6 - y5)
                    # print(longitud3)

                    # aperturaboca
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                    cv2.line(frame, (x7, y7), (x8, y8), (0, 255, 0), t)
                    cv2.circle(frame, (x7, y7), r, (0, 255, 0), cv2.FILLED)
                    cv2.circle(frame, (x8, y8), r, (0, 255, 0), cv2.FILLED)
                    cv2.circle(frame, (cx4, cy4), r, (0, 255, 0), cv2.FILLED)
                    longitud4 = math.hypot(x8 - x7, y8 - y7)
                    # print(longitud4)

                    # muecaizquierda
                    x9, y9 = lista[291][1:]
                    x10, y10 = lista[401][1:]
                    cx5, cy5 = (x9 + x10) // 2, (y9 + y10) // 2
                    cv2.line(frame, (x9, y9), (x10, y10), (255, 0, 0), t)
                    cv2.circle(frame, (x9, y9), r, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (x10, y10), r, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (cx5, cy5), r, (255, 0, 0), cv2.FILLED)
                    longitud5 = math.hypot(x10 - x9, y10 - y9)
                    # print(longitud5)

                    # muecaderecha
                    x11, y11 = lista[61][1:]
                    x12, y12 = lista[177][1:]
                    cx6, cy6 = (x11 + x12) // 2, (y11 + y12) // 2
                    cv2.line(frame, (x11, y11), (x12, y12), (255, 0, 0), t)
                    cv2.circle(frame, (x11, y11), r, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (x12, y12), r, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (cx6, cy6), r, (255, 0, 0), cv2.FILLED)
                    longitud6 = math.hypot(x12 - x11, y12 - y11)
                    # print(longitud6)

            if longitud1 > (int(longitud1calib) * 1.3) and longitud2 > (int(longitud2calib) * 1.3):
#                pwm1.ChangeDutyCycle(9)
                text = 'Cejas Levantadas'
            elif longitud4 > (int(longitud4calib) * 6):
#                pwm1.ChangeDutyCycle(6)
                text = "Boca Abierta"
            elif longitud5 < (int(longitud5calib) * 0.97) and longitud3 > (int(longitud3calib) + 1):
#                pwm2.ChangeDutyCycle(9)
                text = "Mueca Derecha"
            elif longitud6 < (int(longitud6calib) * 0.97) and longitud3 > (int(longitud3calib) + 1):
#                pwm2.ChangeDutyCycle(6)
                text = "Mueca Izquierda"
            elif longitud3 < ((int(longitud3calib) * 0.95)) and longitud4 > (int(longitud4calib) * 1.20):

                text = 'Beso'
            else:
                text = "No hay gesto"
#                pwm1.ChangeDutyCycle(0)
#                pwm2.ChangeDutyCycle(0)

            cv2.putText(frame, text, (60, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, str(round(longitud1)), (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(round(longitud2)), (20, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(round(longitud3)), (20, 110), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(round(longitud4)), (20, 140), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, str(round(longitud5)), (20, 170), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, str(round(longitud6)), (20, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Posicion', frame)
    if cv2.waitKey(50) & 0xFF == 27:
        break
cap.release()
cv2.destroyWindow()
