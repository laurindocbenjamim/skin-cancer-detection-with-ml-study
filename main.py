import os
from pathlib import Path
from src import load_images_from_directory, save_image_2bgr, ImagePreprocessor

# 1. Definir diretórios do projeto
project_directory = Path.cwd()
datasets_path = os.path.join(project_directory, "datasets", "Challenge-tests-2017-images")

# Pastas de saída organizadas por etapa
clean_output_dir = os.path.join(project_directory, "datasets", "1_clean_preprocessed_images")
aug_output_dir = os.path.join(project_directory, "datasets", "2_augmented_train_images")

if not os.path.exists(datasets_path):
    print(f"Pasta não encontrada: {datasets_path}")
else:
    # Passo A: Carregar imagens originais já convertidas para RGB
    images = load_images_from_directory(datasets_path, convert_to_rgb=True)
    print(f"Total de imagens carregadas: {len(images)}")

    # Dicionário para guardar as imagens finais em memória (prontas para o Passo 5: Normalização)
    preprocessed_images = {}

    for name, img_rgb in images.items():
        # 1. Recortar bordas pretas circulares do dermatoscópio
        cropped_img = ImagePreprocessor.crop_black_borders(img_rgb)

        # 2. Redimensionar para o tamanho padrão da rede neuronal (224x224)
        resized_img = ImagePreprocessor.resize_or_downsampling_images(cropped_img, 224, 224)

        # 3. Remover pêlos e artefactos (Black-Hat + Inpainting)
        hairless_img = ImagePreprocessor.remove_hair_from_image(resized_img, name)

        # 4. Correção de iluminação global / constância de cor (Shades of Gray)
        color_constancy_img = ImagePreprocessor.apply_shades_of_gray(hairless_img, power=6)

        # 5. Correção de contraste local (CLAHE)
        final_clean_img = ImagePreprocessor.apply_clahe(color_constancy_img)

        # Guardar a imagem base limpa (serve para Validação/Teste e referência)
        save_image_2bgr(final_clean_img, clean_output_dir, name, prefix="clean")
        preprocessed_images[name] = final_clean_img

        # 6. Data Augmentation (Gera 3 variações por imagem - usar apenas no Treino)
        for i in range(3):
            aug_img = ImagePreprocessor.apply_data_augmentation(final_clean_img)
            save_image_2bgr(aug_img, aug_output_dir, name, prefix=f"aug_{i+1}")

        print(f"Pipeline concluído para: {name}")

    print(f"\nProcessamento concluído! {len(preprocessed_images)} imagens prontas para o Passo 5 (Normalização).")