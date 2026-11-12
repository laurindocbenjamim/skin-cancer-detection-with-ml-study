
import os
import cv2
import numpy as np


def load_images_from_directory(datasets_path):
    """
    Desc: 
        Esta funcção serve para carregar todas imagnes do directorio 'datasets_path'
        de forma eficiente.

    Return:
        Retorna um dicionário onde a chave é o nome do ficheiro e o valor é a imagem em Numpy 
        array
    """

    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp')
    images = {}

    # verificar se o directorio existe
    if(not os.path.exists(datasets_path)):
        print(f"O diretorio {datasets_path} nao foi encontrado.")
        return images

    # percorre os ficheiros do directorio
    for file_name in  os.listdir(datasets_path):
        complete_path = os.path.join(datasets_path, file_name)

        # Vamos ler uma imagem no formato BGR com o comando 'cv2.imread'
        img = cv2.imread(complete_path)

        # verificar de a imagem foi carregada com sucesso, se sim entao adiciona-se a lista de imagens
        if img is not None:
            images[file_name] = img
            print(f"Imagem carregada com sucesso {file_name}.")
        else:
            print(f"Aviso: Nao foi possivel ler o ficheiro {file_name}")
            
    return images
    