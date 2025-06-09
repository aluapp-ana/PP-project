import os
import shutil #copiar arquivos

# Lista de palavras sinalizadas
palavras = [
    "Acontecer", "Aluno", "Amarelo", "America", "Aproveitar",
    "Bala", "Banco", "Banheiro", "Barulho", "Cinco", "Conhecer",
    "Espelho", "Esquina", "Filho", "Maca", "Medo", "Ruim", "Sapo",
    "Vacina", "Vontade"
]
palavras_minusculas = [p.lower() for p in palavras]

# Caminhos
pasta_origem = "D:\dataset_desorganizado"
pasta_destino = "D:\dataset_Organizado"

# Criar pastas de destino para cada palavra
for palavra in palavras_minusculas:
    os.makedirs(os.path.join(pasta_destino, palavra), exist_ok=True)

# Processar sinalizadores
for sinalizador in os.listdir(pasta_origem):
    pasta_sinalizador = os.path.join(pasta_origem, sinalizador)
    pasta_canon = os.path.join(pasta_sinalizador, "Canon")

    if not os.path.isdir(pasta_canon):
        continue

    for video in os.listdir(pasta_canon):
        if video == ".DS_Store":
            continue

        video_path = os.path.join(pasta_canon, video)
        video_lower = video.lower()

        # Tenta identificar a palavra no nome do vídeo
        encontrou = False
        for palavra in palavras_minusculas:
            if palavra in video_lower:
                nome_novo = f"{palavra}_{sinalizador}_{video}"
                destino_path = os.path.join(pasta_destino, palavra, nome_novo)
                shutil.copy(video_path, destino_path)
                encontrou = True
                break

        if not encontrou:
            print(f"[⚠️] Palavra não reconhecida em: {video}")
