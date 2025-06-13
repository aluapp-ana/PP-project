#CODIGO ORGANIZAÇAO FINAL DATA SET PARA TREINAR A MLP 
import os
import shutil
import random
from glob import glob

#Caminho do dataset original (formato do seu código)
CAMINHO_ORIGEM = "D:/16_16_dataset"

#Caminho de saída do novo dataset
CAMINHO_SAIDA = "D:/keypoints_dataset_final"

#Proporções desejadas
PROPORCOES = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

random.seed(42)

#Cria estrutura de saída
for tipo in ["train", "val", "test"]:
    os.makedirs(os.path.join(CAMINHO_SAIDA, tipo), exist_ok=True)

#Percorre cada classe (palavra)
for palavra in sorted(os.listdir(CAMINHO_ORIGEM)):
    caminho_palavra = os.path.join(CAMINHO_ORIGEM, palavra)
    if not os.path.isdir(caminho_palavra):
        continue

    # Cada subpasta é um vídeo (de um sinalizador)
    videos = sorted(os.listdir(caminho_palavra))
    random.shuffle(videos)

    n_total = len(videos)
    n_train = int(n_total * PROPORCOES["train"])
    n_val   = int(n_total * PROPORCOES["val"])
    n_test  = n_total - n_train - n_val

    divisoes = {
        "train": videos[:n_train],
        "val":   videos[n_train:n_train+n_val],
        "test":  videos[n_train+n_val:]
    }

    for tipo, lista_videos in divisoes.items():
        for video_nome in lista_videos:
            caminho_imgs = os.path.join(caminho_palavra, video_nome, "images")
            if not os.path.exists(caminho_imgs):
                continue

            destino_classe = os.path.join(CAMINHO_SAIDA, tipo, palavra)
            os.makedirs(destino_classe, exist_ok=True)

            imagens = glob(os.path.join(caminho_imgs, "*.png"))
            for img_path in imagens:
                nomeimg = f"{palavra}{videonome}{os.path.basename(img_path)}"
                shutil.copy(img_path, os.path.join(destino_classe, nome_img))

print("✅ Dataset reorganizado com sucesso!")