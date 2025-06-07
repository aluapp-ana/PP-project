"""
extrai keypoints de quadros JPG
  - CSVs (coordenadas normalizadas)
  - PNGs (stick-figure com fundo preto)
"""

# imports
import os
import glob
import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# setup do MediaPipe
mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils
mp_face_conn = mp.solutions.face_mesh_connections

holistic = mp_holistic.Holistic(
    static_image_mode=True, # cada quadro é processado separado
    model_complexity=0, # 0 = mais rápido, 1 = mais preciso
    refine_face_landmarks=False, # não usa landmarks refinados
    min_detection_confidence=0.5, # confiança mínima para detecção
)

# usando "mini_mesh" 
USE_MINI_MESH = False

POSE_FACE_IDXS = [0, 2, 5, 9, 10]   # nariz, olho E, olho D, boca E, boca D

# definindo índices dos keypoints do mini-mesh
MINI_MESH_IDXS  = [1, 33, 61, 199, 263, 291] # definindo nariz, olhoE, bocaE, queixo, bocaD, olhoD
MINI_MESH_EDGES = []  # só pontos, sem linhas  (desenho opcional)

# função principal
def processar_pasta_de_quadros(frames_dir, output_root="keypoints_dataset",
                               pad_zeros=5):
    jpg_paths = natsorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not jpg_paths:
        raise ValueError(f"Sem JPG em {frames_dir}")

    seq_name = os.path.basename(os.path.normpath(frames_dir))
    out_imgdir = os.path.join(output_root, seq_name, "images")
    out_csvdir = os.path.join(output_root, seq_name, "keypoints")
    os.makedirs(out_imgdir, exist_ok=True)
    os.makedirs(out_csvdir, exist_ok=True)

    for idx, img_path in enumerate(jpg_paths):
        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            print(f"Falha lendo {img_path}, skiping" )
            continue

        res = holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # coleta os landmarks (keypoints)
        rows = []

        # Pose
        if res.pose_landmarks:
            for i,lm in enumerate(res.pose_landmarks.landmark):
                # Se estamos usando só os 5 anchors, descarta outros pontos da FACE
                if USE_MINI_MESH or (i not in POSE_FACE_IDXS):
                    rows.append(["pose", i, lm.x, lm.y, lm.z, lm.visibility])
                elif not USE_MINI_MESH and i in POSE_FACE_IDXS:
                    rows.append(["face_pose", i, lm.x, lm.y, lm.z, lm.visibility])

        # Mãos
        if res.left_hand_landmarks:
            for i,lm in enumerate(res.left_hand_landmarks.landmark):
                rows.append(["left_hand", i, lm.x, lm.y, lm.z, lm.visibility])
        if res.right_hand_landmarks:
            for i,lm in enumerate(res.right_hand_landmarks.landmark):
                rows.append(["right_hand", i, lm.x, lm.y, lm.z, lm.visibility])

        # usando o Mini-mesh definido lá em cima
        if USE_MINI_MESH and res.face_landmarks:
            for i in MINI_MESH_IDXS:
                lm = res.face_landmarks.landmark[i]
                rows.append(["face_mesh", i, lm.x, lm.y, lm.z, None])

        if not rows:
            print(f"Nenhum keypoint em {img_path}")
            continue

        # salvando o CSV com os keypoints
        pd.DataFrame(rows, columns=["segment","id","x","y","z","visibility"]) \
          .to_csv(os.path.join(out_csvdir, f"frame_{idx:0{pad_zeros}d}.csv"),
                  index=False)

        # desenhando o stick-figure
        h, w = frame_bgr.shape[:2]
        black = np.zeros((h, w, 3), dtype=np.uint8)

        # Corpo + mãos
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(black, res.pose_landmarks,
                                      mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec((255,255,255),2,2),
                                      mp_drawing.DrawingSpec((255,255,255),2))
        for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
            if hand:
                mp_drawing.draw_landmarks(black, hand,
                                          mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec((255,255,255),2,2),
                                          mp_drawing.DrawingSpec((255,255,255),2))

        # Mini-mesh (só pontos)
        if USE_MINI_MESH and res.face_landmarks:
            reduced = landmark_pb2.NormalizedLandmarkList(
                landmark=[res.face_landmarks.landmark[i] for i in MINI_MESH_IDXS])
            mp_drawing.draw_landmarks(
                black, reduced, MINI_MESH_EDGES,
                mp_drawing.DrawingSpec((255,255,255),2,2),
                mp_drawing.DrawingSpec((255,255,255),2))

        cv2.imwrite(os.path.join(out_imgdir, f"frame_{idx:0{pad_zeros}d}.png"),
                    black)
        print(f"{seq_name} - quadro {idx+1}/{len(jpg_paths)}")

    print(f"'{seq_name}' concluído -> {out_imgdir}")

# uso
if __name__ == "__main__":
    frames_pasta = "/D-SSD/mussi/graduacao/frames_dataset"
    processar_pasta_de_quadros(frames_pasta, output_root="keypoints_dataset")