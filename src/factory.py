
import os
import cv2
import numpy as np

# Funcao para visualizar imagens
def display_image(image, title='Imagem original'):
        """
        Desc:
            - Funcao usada para exibir uma imagem
        """
        if image is not None:
            # 1. Criar uma cópia para não alterar a imagem original do processamento
            img_com_texto = image.copy()
            
            # 2. Se a imagem estiver em tons de cinza (2D), converter para BGR para o texto ficar colorido
            if len(img_com_texto.shape) == 2:
                img_com_texto = cv2.cvtColor(img_com_texto, cv2.COLOR_GRAY2BGR)
                
            # 3. Escrever o texto na parte inferior da imagem
            texto = "Pressiona qualquer tecla..."
            cv2.putText(
                img_com_texto, 
                texto, 
                (10, 210),                   # Posição (X, Y) perto do fundo da imagem 224x224
                cv2.FONT_HERSHEY_SIMPLEX,    # Tipo de letra
                0.4,                         # Tamanho da letra (pequeno para caber em 224px)
                (0, 255, 0),                 # Cor em BGR (Verde brilhante para destacar)
                1,                           # Espessura da linha
                cv2.LINE_AA                  # Suavização da letra
            )
            
            cv2.imshow(f"Imagem OpenCV - {title}", img_com_texto)

            #print("Preciona qualquer tecla na tela da imagem para continuar...")
            #cv2.waitKey(0) # Espera ate o utilizador precionar uma tecla
            #cv2.destroyAllWindows() # Fecha a janela aberta
def closeDisplay():
        print("Preciona qualquer tecla na tela da imagem para continuar...")
        cv2.waitKey(0) # Espera ate o utilizador precionar uma tecla
        cv2.destroyAllWindows() # Fecha a janela aberta

# Function to load the files
def load_images_from_directory(datasets_path, convert_to_rgb=True):
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
            img_bgr = cv2.imread(complete_path)
            
            # Validar a extensão das imagens
            image_is_valid = file_name.lower().endswith(extensoes_validas)

            if not image_is_valid:
                print(f"Arquivo inválido [{file_name}]. Formatos permitidos {extensoes_validas}")
                continue
            
            # verificar de a imagem foi carregada com sucesso, se sim entao adiciona-se a lista de imagens
            if img_bgr is None:
                print(f"Aviso: Nao foi possivel ler o ficheiro {file_name}")
                continue

            # Agora vamos converter as imagens de BGR para o padrao RGB to fit with the ML and CNN models
            # Razoes: Compatibilidade com Modelos de Deep Learning (Transfer Learning), Visualização Correta (Matplotlib / Jupyter Notebooks), Análise Clínica e Biomarcadores (Critérios ABCD)
            if convert_to_rgb:
                final_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                final_img = img_bgr

            images[file_name] = final_img
            #print(f"Imagem carregada com sucesso {file_name}")
                    
                
        return images

# Function to inspect the images
def inspect_dimentions(images_dict):
        """
        Desc:
            This function is used to inspect the images dimentions, data types and channels
        Return:
            None
        """

        for name, img in images_dict.items():
            # img.shape() devolve a altura, largura e canais para imagens coloridas (identificado pelo 3)
            if len(img.shape) == 3:
                height,width,channels = img.shape
            else:
                height,width = img.shape
                channels = 1 # Imagem em escala de cinzento

            print(f"--- File {name} ---")
            print(f"  - Shape (formato completo): {img.shape}")
            print("")

# Guardar a imagem processada (Converter de volta para BGR para o OpenCV guardar corretamente)
def save_image_2bgr(img_sem_pelos_rgb, output_dir, image_name, prefix="cleaned"):
        """
        Desc:
            Esta funcao serve para guardar a imagem processada no diretorio de saida especificado.
        Return:
            None
        """
        if img_sem_pelos_rgb is None:
            print("Imagem inválida fornecida para salvar.")
            return

        if output_dir is None or not os.path.exists(output_dir):
            mkdir_status = os.makedirs(output_dir, exist_ok=True)
            if mkdir_status is None:
                print(f"Diretório de saída criado: {output_dir}")
            else:
                print(f"Falha ao criar diretório de saída: {output_dir}")
                return

        # Guardar a imagem processada (Converter de volta para BGR para o OpenCV guardar corretamente)
        final_image_bgr = cv2.cvtColor(img_sem_pelos_rgb, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(output_dir, f"{prefix}_{image_name}")
        
        # Salvar a imagem processada
        cv2.imwrite(output_path, final_image_bgr)

        return final_image_bgr

  