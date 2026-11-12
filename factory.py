
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

    