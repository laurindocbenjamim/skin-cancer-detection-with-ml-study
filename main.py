
import os
from pathlib import Path

from src import load_images_from_directory

# Obter o caminho do directorio de dados de forma eficiente sem caminho absoluto com o Pathlib
# obter a pasta onde a pipeline esta a ser executada
project_directory = Path.cwd()

# Pegar o caminho do directorio raiz do projecto
# root_directory = project_directory.parent

datasets_path = os.path.join(project_directory, "datasets", "Challenge-tests-2017-images")

if not os.path.exists(datasets_path):
    print(f"Pasta nao encontrada {datasets_path}")
else:
    images = load_images_from_directory(datasets_path)
    print('Image List')
    print(len(images))