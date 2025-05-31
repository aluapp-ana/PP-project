import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np

# inicializa o mediapipe holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# define pontos de interesse para o rosto
boca_indices = list(range(61, 89))
sobrancelha_esq_indices = list(range(55, 66))
sobrancelha_dir_indices = list(range(285, 296))
olho_esq_indices = list(range(133, 145)) + list(range(145, 165))
olho_dir_indices = list(range(362, 374)) + list(range(374, 394))

face_indices = (boca_indices + sobrancelha_esq_indices +
                sobrancelha_dir_indices + olho_esq_indices + olho_dir_indices)

def extrair_e_processar_frames(video_path, output_folder, num_frames=32, espacamento=15):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    frames_processados = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_images = os.path.join(output_folder, video_name, "images")
    output_csvs = os.path.join(output_folder, video_name, "keypoints")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_csvs, exist_ok=True)

    while frames_processados < num_frames and cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # converte BGR para RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        keypoints_data = []

        # função auxiliar para extrair keypoints
        def extrair_landmarks(landmarks, label):
            data = []
            for i, lm in enumerate(landmarks.landmark):
                data.append([label, i, lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else None])
            return data

        # extrai e armazena keypoints do corpo
        if results.pose_landmarks:
            keypoints_data += extrair_landmarks(results.pose_landmarks, "pose")

        # mãos
        if results.left_hand_landmarks:
            keypoints_data += extrair_landmarks(results.left_hand_landmarks, "left_hand")

        if results.right_hand_landmarks:
            keypoints_data += extrair_landmarks(results.right_hand_landmarks, "right_hand")

        # rosto filtrado
        if results.face_landmarks:
            for i in face_indices:
                lm = results.face_landmarks.landmark[i]
                keypoints_data.append(["face", i, lm.x, lm.y, lm.z, None])

        if keypoints_data:
            # salva os keypoints em CSV
            df = pd.DataFrame(keypoints_data, columns=["segment", "id", "x", "y", "z", "visibility"])
            csv_path = os.path.join(output_csvs, f"frame_{frames_processados:02d}.csv")
            df.to_csv(csv_path, index=False)

            # cria imagem preta e desenha todos os keypoints
            img_height, img_width = frame.shape[:2]
            black_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            # desenha as landmarks na imagem preta
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(black_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(black_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(black_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(black_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

            # salva a imagem
            img_path = os.path.join(output_images, f"frame_{frames_processados:02d}.jpg")
            cv2.imwrite(img_path, black_image)

            frames_processados += 1
            print(f"Frame {frames_processados}/{num_frames} processado.")

        frame_idx += espacamento

    cap.release()
    print("Processamento finalizado.")

# caminho
video = "C:/Users/anapa/OneDrive - PUCRS - BR/Documentos/dataset_KEYPOINTS/videoteste.mp4"
saida = "saida_frames"
extrair_e_processar_frames(video, saida)






######################################################################################################################



#video = cv2.VideoCapture(0)

#hands = mp.solutions.hands
#Hands = hands.Hands(max_num_hands=3)
#mpDwaw = mp.solutions.drawing_utils

#while True:
#    success, img = video.read()
#    frameRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#    results = Hands.process(frameRGB)
#    handPoints = results.multi_hand_landmarks
#    h, w, _ = img.shape
#    pontos = []
#    if handPoints:
#        for points in handPoints:
#            mpDwaw.draw_landmarks(img, points,hands.HAND_CONNECTIONS)
#            #podemos enumerar esses pontos da seguinte forma
#            for id, cord in enumerate(points.landmark):
#               cx, cy = int(cord.x * w), int(cord.y * h)
#                #cv2.putText(img, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                pontos.append((cx,cy))
#
#            dedos = [8,12,16,20]
#            contador = 0
#            if pontos:
#                if pontos[4][0] < pontos[3][0]:
#                    contador += 1
#                for x in dedos:
#                   if pontos[x][1] < pontos[x-2][1]:
#                       contador +=1

#            cv2.rectangle(img, (80, 10), (200,110), (255, 0, 0), -1)
#            cv2.putText(img,str(contador),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)
#            #print(contador)

#    cv2.imshow('Imagem',img)
#    cv2.waitKey(1)



#cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)


# for id,cord in enumerate(points.landmark):
#     cx, cy = int(cord.x * w), int(cord.y * h)
#     cv2.putText(img, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     pontos.append([cx,cy])
#     if pontos:
#         print(pontos)