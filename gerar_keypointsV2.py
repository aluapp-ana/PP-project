# imports
import os
import glob
import cv2
import numpy as np
from natsort import natsorted
import mediapipe as mp

# Configurações principais
CAMINHO_VIDEOS = "D:/dataset_Organizado"
CAMINHO_SAIDA  = "D:/keypoints_dataset"

# setup do MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=True,       # cada quadro é processado separado
    model_complexity=0,           # 0 = mais rápido, 1 = mais preciso
    refine_face_landmarks=False,  # não usa landmarks refinados
    min_detection_confidence=0.5, # confiança mínima para detecção
)

# usando "mini_mesh"
USE_MINI_MESH = False

POSE_FACE_IDXS = [0, 2, 5, 9, 10]   # nariz, olho E, olho D, boca E, boca D
MINI_MESH_IDXS  = [1, 33, 61, 199, 263, 291]
MINI_MESH_EDGES = []  # só pontos, sem linhas

def extrair_frames_do_video(video_path, dir_saida_frames):
    os.makedirs(dir_saida_frames, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_path = os.path.join(dir_saida_frames, f"frame_{idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()
    return idx

def gerar_keypoints(frames_dir, output_dir, pad_zeros=5):
    jpg_paths = natsorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not jpg_paths:
        print(f" Sem JPG em {frames_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for idx, img_path in enumerate(jpg_paths):
        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            print(f" Erro ao ler {img_path}")
            continue

        res = holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        h, w = frame_bgr.shape[:2]
        black = np.zeros((h, w, 3), dtype=np.uint8)

        # Pose
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                black,
                res.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )

        # Mãos
        for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
            if hand:
                mp_drawing.draw_landmarks(
                    black,
                    hand,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )

        # Rosto (com pontos maiores e uniformes)
        if res.face_landmarks:
            mp_drawing.draw_landmarks(
                black,
                res.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2)
            )

        out_path = os.path.join(output_dir, f"frame_{idx:0{pad_zeros}d}.png")
        cv2.imwrite(out_path, black)

    print(f" {len(jpg_paths)} quadros processados em {output_dir}")

def processar_todos_os_videos():
    for palavra in os.listdir(CAMINHO_VIDEOS):
        pasta_palavra = os.path.join(CAMINHO_VIDEOS, palavra)
        if not os.path.isdir(pasta_palavra):
            continue

        for video_nome in os.listdir(pasta_palavra):
            if not video_nome.endswith(".mp4"):
                continue

            caminho_video = os.path.join(pasta_palavra, video_nome)
            nome_base = os.path.splitext(video_nome)[0]

            dir_frames_temp = os.path.join("temp_frames", palavra, nome_base)
            dir_output_png = os.path.join(CAMINHO_SAIDA, palavra, nome_base, "images")

            print(f"\n Processando vídeo: {video_nome}")
            qtd_frames = extrair_frames_do_video(caminho_video, dir_frames_temp)
            if qtd_frames == 0:
                print(f" Nenhum quadro extraído de {video_nome}")
                continue

            gerar_keypoints(dir_frames_temp, dir_output_png)

    # Limpa os quadros temporários (opcional)
    import shutil
    shutil.rmtree("temp_frames", ignore_errors=True)

# Executar
if __name__ == "__main__":
    processar_todos_os_videos()
