from ctypes import util
import os
import cv2
import numpy as np
import albumentations as A
from .factory import display_image

class ImagePreprocessor:

    def __init__(self):
        pass    

    # Funcao para redimencionar as imagens 
    @staticmethod
    def resize_or_downsampling_images(img, new_width=224, new_height=224):
        """
        Desc:
            Funcao para redimencionar as imagens para dimensoes fixas. 
            utiliza a funcao cv2.INTER_AREA que e apropriada/ideal para reduzir imagens (shrink/downsampling)
            Devemos redimencionar pelos motivos abaixo:
            - Exigência de Dimensão Fixa nas Redes Neuronais (CNNs)
            - Gestão de Memória e Eficiência Computacional (GPU/VRAM)
            - Padronização de Fontes de Aquisição
        Return:
            Resized image
        """

        if img is None:
            print(f"Imagem invalida {img}")
            return None
        if not isinstance(new_width, int) and isinstance(new_height, int):
            print(f"Tipos de dados invalidos w:{new_height}/h:{new_height}")
            return None

        dimensions = (new_width, new_height)
        image_resized = cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)

        return image_resized

    # Implementação da Remoção de Pêlos (Black-Hat + Inpainting)
    # Converter para escala de cinzentos para a operação morfológica

    @staticmethod
    def remove_hair_from_image(resized_image, image_name = None):
        """
        Desc:
            Esta funcao serve para remover os pelos das imagens de pele utilizando a tecnica de Black-Hat + Inpainting.
            A remoção de pelos é uma etapa importante no pré-processamento de imagens dermatológicas, pois os pelos podem interferir na análise da lesão cutânea, afetando a precisão dos modelos de aprendizado de máquina.
        
            A operação Black-Hat extrai elementos que são mais escuros do que a sua vizinhança direta (perfeito para pêlos escuros sobre a pele). O tamanho do kernel define a "janela" de análise.
            O valor 9 sao valores do kernel que definem qual tamnaho de pelo detectar. 
            Valores menores definem pelos microscopicos, maiores=macroscopicos mas kernel grande como 21 pode pode confundir as bordas da lesao 
        Return:
            Retorna a imagem sem pelos
        """

        if resized_image is None:
            print("Imagem inválida fornecida para remoção de pelos.")
            return None

        #display_image(resized_image, f"Imagem original - {image_name}")
        # converter a imagem para escala de cinza porque a operação morfológica Black-Hat é aplicada em imagens de um único canal
        img_gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

        #display_image(img_gray, "Imagem em tom de Gray")
        # Definir o kernel para a operacao morfologica Black-Hat. 
        # O tamanho do kernel pode ser ajustado dependendo da densidade e espessura dos pelos na imagem.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) 

        # Aplicar o Black-hat
        blackhat_img = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

        #display_image(blackhat_img, f"Imagem com blackhat aplicada - {image_name}")
        # Binarizar a imagem para isolar os pelos pretos, convertendo a imagem cinza do blackhat 
        # numa mascara puramente Preto e Branco (Binaria), decidindo o que e considerado Pelo e o que e ignorado
        _, tresh_img = cv2.threshold(blackhat_img, 20, 255, cv2.THRESH_BINARY)

        #display_image(tresh_img, f"Imagem com Treshold de 40/255 aplicada - {image_name}")

        tresh_image_filtered = ImagePreprocessor.filter_images_by_area(tresh_img)

        #display_image(tresh_image_filtered, f"Imagem com Treshold filtrada aplicada - {image_name}")

        # Dilatar ligeiramente a mascara para garantir cobertura completa das bordas do pelo
        kernel_dilatation = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask_image = cv2.dilate(tresh_image_filtered, kernel_dilatation, iterations = 1)

        #display_image(mask_image, f"Imagem com mask_image dilatada - {image_name}")

        # Inpainting(reconstrução da imagem mascarada)
        # Nota: O cv2.inpaint espera imagens em BGR se formos gravar com cv2.imwrite,
        # por isso convertemos temporariamente ou aplicamos em RGB e convertemos no fim.
        img_sem_pelos_rgb = cv2.inpaint(resized_image, mask_image, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        #display_image(img_sem_pelos_rgb, f"Imagem com inpainting aplicado - {image_name}")

        return img_sem_pelos_rgb


    # Filtrar por area ou comprimento da imagem, para remover imagens muito pequenas ou muito grandes
    #Como os pelos são estruturas alongadas e o ruído da pele são apenas pontos isolados, podes adicionar uma verificação de tamanho/área para apagar da máscara tudo o que seja demasiado pequeno para ser um pelo:
    @classmethod
    def filter_images_by_area(cls, images_tresh):
        """
        Desc:
            Esta funcao serve para filtrar a imagem binaria (tresh) por area, removendo pequenos ruídos e mantendo apenas os contornos significativos (como pelos).
        Return:
            Retorna a imagem binaria filtrada
        """

        # Encontrar contornos na máscara
        num_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(images_tresh, connectivity=8)

        # Criar uma nova máscara para armazenar os contornos filtrados
        cleaned_mask = np.zeros_like(images_tresh)
        for i in range(1, num_labels):  # Começar de 1 para ignorar o fundo
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Condição: Um pelo real tende a ser comprido (uma das dimensões é significativamente maior que a outra)
            # e tem uma área mínima. Ajusta estes limites conforme necessário.
            if area > 15 and (width > 10 or height > 10):  # Ajustar o limiar de área e dimensão conforme necessário
                cleaned_mask[labels == i] = 255

        return cleaned_mask

    # Fucncao para aplicar o CLAHE (Contrast Limited Adaptive Histogram Equalization) para melhorar o contraste da imagem, especialmente útil em imagens médicas.
    @staticmethod
    def apply_clahe(rgb_image):
        """
        Desc:
            Esta funcao aplica o CLAHE (Contrast Limited Adaptive Histogram Equalization) para melhorar o contraste da imagem.
            Ao aplicar o CLAHE, o contraste da imagem é melhorado, especialmente em regiões de baixa luminosidade, 
            o que pode ajudar a destacar características importantes da lesão cutânea.
        Return:
            Retorna a imagem com contraste melhorado
        """
        if rgb_image is None:
            print("Imagem inválida fornecida para aplicar CLAHE.")
            return None

        # 1. Converter a imagem de RGB para o espaço de cores LAB
        # O canal L (Lightness) contém toda a informação de brilho/luminância separada da cor
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(lab_image)

        # inicializar o objecto CLAHE
        # O clipLimit controla o limite do contraste (evita saturacao excessiva) e tileGridSize define o tamanho da região para a equalização adaptativa
        # O canal lightness contem toda informacao do brilho/luminancia separada da cor, por isso aplicamos CLAHE apenas neste canal
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)

        # Juntar o canal L corrigido de volta com o canal A e B originais
        lab_merged = cv2.merge((l_clahe, a, b))

        # Converter de volta para o espaco de cores RGB
        rgb_clahe = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)

        return rgb_clahe

    # Funcao para aplicar shades of gray para melhorar o contraste da imagem, especialmente útil em imagens médicas.
    @staticmethod
    def apply_shades_of_gray(rgb_image, power=6):
        """
        Desc:
            Aplica o algoritmo Shades of Gray preservando a luminosidade média original
            para evitar saturação (estouramento de branco) em peles claras.
        Parameters:
            rgb_image: Imagem RGB de entrada
            power: Potência para o cálculo da média ponderada (default=6) segundo a. norma de Minkoski (p6 que e o padrao na literatura ISIC)
        Return:
            Retorna a imagem com constancia de cor melhorada
        """
        if rgb_image is None:
            print("Imagem inválida fornecida para aplicar Shades of Gray.")
            return None
        # Converter a imagem para float32 para evitar ovverflow durante as potencias 
        image_float = rgb_image.astype(np.float32)

        # Calcular a norma de Minkoski para cada canal (R,G,B)
        # Soma os valores elevados a 'power' e tira a raiz de indice 'power' para cada canal
        normas_canais = np.power(np.mean(np.power(image_float, power), axis=(0,1)), 1/power)

        # Calcular a norma euclidiana (L2) do vetor de iluminacao estimado
        normas_iluminacao = np.sqrt(np.sum(np.power(normas_canais, 2)))

        # Evitar divisao por Zero
        normas_canais = np.where(normas_canais == 0, 1e-6, normas_canais)

        # calcular os fatores de escala para equilibrar os 3 canais (R,G,B)
        fatores_escala = normas_iluminacao / (normas_canais * np.sqrt(3))

        # Aplicando a correcao multiplicando cada canal com o seu fator de escala correspondente
        image_corrected = image_float * fatores_escala

        # Limitar os valores aos intervalos validos [0, 255] e converter de volta para unit8
        image_corrigida = np.clip(image_corrected, 0, 255).astype(np.uint8)

        return image_corrigida

    # Redimensionar as bordas pretas da imagem para remover áreas desnecessárias e focar na região de interesse (ROI).
    @staticmethod
    def crop_black_borders(img_rgb, black_threshold=15, security_margin=5):
        """
        Desc:
            Deteta e remove bordas escuras/circulares típicas de dermatoscópios,
            mantendo apenas a região útil da pele e da lesão.
        Parameters:
            rgb_image: Imagem de entrada no formato RGB.
            black_threshold: Limiar de intensidade abaixo do qual o píxel é considerado borda preta.
            margin: Pequena margem interna (em píxeis) para limpar resíduos da borda curva.
        Return:
            Imagem recortada no formato RGB.
        """
        if img_rgb is None:
            print("Imagem inválida fornecida para remover bordas pretas.")
            return None

        height_origin, width_origin = img_rgb.shape[:2]

        # Converter a imagem para escala de cinza para identificar as bordas pretas
        image_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # criar a mascara binanria para remover as bordas pretas, onde os pixels abaixo do limiar são considerados borda preta
        # Cria imagem iluminada (branca) e as bordas pretas tornam-se pretas na máscara binária
        _, thresh_image = cv2.threshold(image_gray, black_threshold, 255, cv2.THRESH_BINARY)

        # Encontrar os contornos da area iluminada (branca) na imagem binária
        contours,_ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("Nenhum contorno encontrado para remover bordas pretas.")
            return img_rgb  # Retorna a imagem original se não houver contornos

        # Selecionar o maior contorno (Janela principal do Dermatoscópio)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # Evitar corte se a area util for anormalmente pequena
        util_area = w * h
        total_area = height_origin * width_origin

        if util_area < 0.2 * total_area:
            return img_rgb

        # Aplicar margem de segurança sem ultrapassar os limites da imagem
        x1 = max(0, x + security_margin)
        y1 = max(0, y + security_margin)
        x2 = min(width_origin, x + w - security_margin)
        y2 = min(height_origin, y + h - security_margin)

        # Garantir que as cooredenadas finais são validas
        if x2 <= x1 or y2 <= y1:
            return img_rgb

        # Efetuar o recorte na imagem RGB original
        cropped_image = img_rgb[x1:x2, y1:y2]

        return cropped_image

    # Applicar o processo de Argumentation com o algoritimo Albumination
    @staticmethod
    def apply_data_augmentation(rgb_image):
        """
        Desc:
            Aplica tranformacoes seguras de Data Argumentation usando o Albuminations
            para aumentar a variabilidade de imagens dermatoscopicas sem alterar a natureza clinica da lesao.
        Parameters:
            rgb_image: Imagem de entrada no formato RGB (uint8).
        Return:    
               Imagem transformada no formato RGB (uint8).      
        """
        if rgb_image is None:
            print("Imagem fornecida inválida para Data Augmentation.")
            return None

        # Definir o pipeline de transformações seguras para imagens dermatoscópicas
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1), # Zoom leve entre 90% e 110%
                translate_percent=(-0.05, 0.05), # Deslocamento maximo de 5% da largura/altura
                rotate=(-45, 45), # Rotacao maximo de 45 graus
                mode=cv2.BORDER_REFLECT_101, # Preenchimento refletido (espelhamento das bordas) para evitar bordas pretas
                p=0.7
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, # Variacao subtil de brilho (+/- 10%)
                contrast_limit=0.1,  # Variacao de contraste (+/- 10%)
                p=0.5
            )
        ])

        # Aplicar a transformacao (a Albumination espera e devolve o dicionario)
        augmented = transform(image=rgb_image)
        return augmented["image"]


    # Funcao para normalizacao das imagens
    @staticmethod
    def normalize_image(rgb_image, method="imagenet"):
        """
        Desc:
            Normaliza os valores dos pixeis da imagem RGB (uint8 [0, 255]) para float32,
            preparando o Tensor Flow para entrada em Redes Neuronais Convolucionais (CNNs).
        Parameters:
            rgb_image: imagem de entrada no formato RGB (uint8)
            method: 'minmax' para intervalo [0, 1] ou 'imagenet' para padronizacao Z-Score
        """
        