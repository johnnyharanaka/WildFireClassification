"""
Converte o dataset NWPU VHR-10 para o formato COCO

Entrada (NWPU format):
- positive image set/ (imagens)
- ground truth/ (arquivos .txt com anotações)
  Formato por linha: (x1,y1),(x2,y2),class_id

Saída (COCO format):
data/
├── Train/
│   ├── images/
│   │   ├── 001.jpg
│   │   └── ...
│   └── annotations.json
├── Val/
│   ├── images/
│   └── annotations.json
└── Test/
    ├── images/
    └── annotations.json

annotations.json segue o formato COCO:
{
    "images": [{"id": 1, "file_name": "001.jpg", "width": 800, "height": 600}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "area": w*h}],
    "categories": [{"id": 1, "name": "airplane", "supercategory": "object"}]
}
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict
from PIL import Image
from core.config.config import log_info

# Mapeamento oficial NWPU VHR-10 (class_id -> nome)
CLASS_MAPPING = {
    1: "airplane",
    2: "ship",
    3: "storage_tank",
    4: "baseball_diamond",
    5: "tennis_court",
    6: "basketball_court",
    7: "ground_track_field",
    8: "harbor",
    9: "bridge",
    10: "vehicle"
}


def parse_nwpu_annotation(annotation_line: str) -> Tuple[int, int, int, int, int]:
    """
    Parse uma linha de anotação NWPU

    Formato: (x1,y1),(x2,y2),class_id

    Returns:
        Tuple (x1, y1, x2, y2, class_id) ou None se inválido
    """
    try:
        # Remover parênteses e espaços
        line = annotation_line.strip().replace('(', '').replace(')', '')
        parts = [p.strip() for p in line.split(',')]

        if len(parts) >= 5:
            x1, y1, x2, y2, class_id = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            return x1, y1, x2, y2, class_id
    except (ValueError, IndexError) as e:
        log_info(f"AVISO: Erro ao parsear linha '{annotation_line}': {e}")

    return None


def load_nwpu_annotations(gt_dir: str, images_dir: str) -> Dict[str, List[Tuple]]:
    """
    Carrega todas as anotações NWPU

    Returns:
        Dict: {image_name: [(x1, y1, x2, y2, class_id), ...]}
    """
    image_annotations = defaultdict(list)

    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]

    for gt_file in gt_files:
        image_name = gt_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_name)

        # Verificar se a imagem existe
        if not os.path.exists(image_path):
            continue

        gt_path = os.path.join(gt_dir, gt_file)

        try:
            with open(gt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parsed = parse_nwpu_annotation(line)
                    if parsed:
                        image_annotations[image_name].append(parsed)
        except Exception as e:
            log_info(f"AVISO: Erro ao ler {gt_file}: {e}")
            continue

    return dict(image_annotations)


def split_dataset_stratified(image_annotations: Dict[str, List[Tuple]],
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             seed: int = 42) -> Tuple[Dict, Dict, Dict]:
    """
    Divide o dataset mantendo distribuição de classes

    Estratégia: agrupa por classe principal (mais frequente na imagem)
    """
    random.seed(seed)

    # Agrupar imagens por classe principal
    class_to_images = defaultdict(list)

    for image_name, annotations in image_annotations.items():
        if not annotations:
            continue

        # Encontrar classe mais frequente na imagem
        class_counts = defaultdict(int)
        for _, _, _, _, class_id in annotations:
            class_counts[class_id] += 1

        main_class = max(class_counts.items(), key=lambda x: x[1])[0]
        class_to_images[main_class].append(image_name)

    train_split = {}
    val_split = {}
    test_split = {}

    # Dividir cada classe proporcionalmente
    for class_id, images in class_to_images.items():
        random.shuffle(images)
        n_images = len(images)

        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # Adicionar anotações completas
        for img in train_images:
            train_split[img] = image_annotations[img]
        for img in val_images:
            val_split[img] = image_annotations[img]
        for img in test_images:
            test_split[img] = image_annotations[img]

        log_info(f"  {CLASS_MAPPING[class_id]:20s}: Train={len(train_images):3d}, Val={len(val_images):3d}, Test={len(test_images):3d}")

    return train_split, val_split, test_split


def create_coco_annotations(split_data: Dict[str, List[Tuple]],
                           images_dir: str,
                           split_name: str) -> Dict:
    """
    Cria o dicionário de anotações no formato COCO

    Args:
        split_data: {image_name: [(x1, y1, x2, y2, class_id), ...]}
        images_dir: Diretório com as imagens originais
        split_name: Train, Val ou Test

    Returns:
        Dict no formato COCO
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Criar categorias
    for class_id, class_name in CLASS_MAPPING.items():
        coco_format["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "object"
        })

    annotation_id = 1
    image_id = 1

    for image_name in sorted(split_data.keys()):
        image_path = os.path.join(images_dir, image_name)

        # Obter dimensões da imagem
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            log_info(f"AVISO: Erro ao ler imagem {image_name}: {e}")
            continue

        # Adicionar informação da imagem
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # Adicionar anotações desta imagem
        for x1, y1, x2, y2, class_id in split_data[image_name]:
            # Converter para formato COCO (x, y, width, height)
            # NWPU usa coordenadas (x1, y1) top-left e (x2, y2) bottom-right
            bbox_x = x1
            bbox_y = y1
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height

            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                "area": bbox_area,
                "iscrowd": 0
            })

            annotation_id += 1

        image_id += 1

    return coco_format


def copy_images_and_save_annotations(images_dir: str,
                                     split_data: Dict[str, List[Tuple]],
                                     output_dir: str,
                                     split_name: str):
    """
    Copia imagens e salva annotations.json no formato COCO
    """
    # Criar diretório de imagens
    images_output_dir = os.path.join(output_dir, split_name, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    # Copiar imagens
    log_info(f"\n{split_name}: Copiando {len(split_data)} imagens...")
    for image_name in split_data.keys():
        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(images_output_dir, image_name)

        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            log_info(f"AVISO: Erro ao copiar {image_name}: {e}")

    # Criar anotações COCO
    log_info(f"{split_name}: Criando annotations.json...")
    coco_annotations = create_coco_annotations(split_data, images_dir, split_name)

    # Salvar annotations.json
    annotations_path = os.path.join(output_dir, split_name, "annotations.json")
    with open(annotations_path, 'w') as f:
        json.dump(coco_annotations, f, indent=2)

    # Estatísticas
    num_images = len(coco_annotations["images"])
    num_annotations = len(coco_annotations["annotations"])
    avg_annotations = num_annotations / num_images if num_images > 0 else 0

    log_info(f"{split_name}: Salvo {num_images} imagens, {num_annotations} anotações (avg: {avg_annotations:.1f} boxes/img)")


def convert_nwpu_to_coco(nwpu_root_dir: str,
                        output_dir: str = "data",
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        seed: int = 42):
    """
    Converte dataset NWPU VHR-10 para formato COCO
    """
    log_info("="*70)
    log_info("Convertendo NWPU VHR-10 para formato COCO")
    log_info("="*70)

    # Caminhos do dataset original
    images_dir = os.path.join(nwpu_root_dir, "positive image set")
    gt_dir = os.path.join(nwpu_root_dir, "ground truth")

    # Verificar se existem
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {images_dir}")
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"Diretório de ground truth não encontrado: {gt_dir}")

    log_info(f"\nDataset NWPU encontrado em: {nwpu_root_dir}")
    log_info(f"  Imagens: {images_dir}")
    log_info(f"  Anotações: {gt_dir}\n")

    # Carregar anotações
    log_info("Carregando anotações NWPU...")
    image_annotations = load_nwpu_annotations(gt_dir, images_dir)

    total_images = len(image_annotations)
    total_boxes = sum(len(anns) for anns in image_annotations.values())
    avg_boxes = total_boxes / total_images if total_images > 0 else 0

    log_info(f"{total_images} imagens com {total_boxes} bounding boxes (avg: {avg_boxes:.1f} boxes/img)\n")

    # Dividir dataset
    log_info("Dividindo dataset em Train/Val/Test...")
    log_info(f"Ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}\n")

    train_split, val_split, test_split = split_dataset_stratified(
        image_annotations,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # Processar cada split
    for split_name, split_data in [("Train", train_split), ("Val", val_split), ("Test", test_split)]:
        if split_data:
            copy_images_and_save_annotations(images_dir, split_data, output_dir, split_name)

    log_info("\n" + "="*70)
    log_info("Conversão concluída com sucesso!")
    log_info("="*70)
    log_info(f"\nEstrutura de saída (formato COCO):")
    log_info(f"{output_dir}/")
    log_info(f"├── Train/")
    log_info(f"│   ├── images/          ({len(train_split)} imagens)")
    log_info(f"│   └── annotations.json")
    log_info(f"├── Val/")
    log_info(f"│   ├── images/          ({len(val_split)} imagens)")
    log_info(f"│   └── annotations.json")
    log_info(f"└── Test/")
    log_info(f"    ├── images/          ({len(test_split)} imagens)")
    log_info(f"    └── annotations.json")
    log_info(f"\nDiretório: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Converter NWPU VHR-10 para formato COCO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplo de uso:
  python convert_nwpu_to_coco.py --nwpu-dir /path/to/NWPU-VHR-10 --output-dir data_nwpu

Estrutura esperada do NWPU:
  NWPU-VHR-10/
  ├── positive image set/  (imagens .jpg)
  └── ground truth/        (anotações .txt)

Estrutura de saída (COCO):
  data_nwpu/
  ├── Train/
  │   ├── images/
  │   └── annotations.json
  ├── Val/
  │   ├── images/
  │   └── annotations.json
  └── Test/
      ├── images/
      └── annotations.json
        """
    )

    parser.add_argument(
        "--nwpu-dir",
        type=str,
        required=True,
        help="Diretório raiz do dataset NWPU VHR-10"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_nwpu",
        help="Diretório de saída (padrão: data_nwpu)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proporção para treino (padrão: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proporção para validação (padrão: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proporção para teste (padrão: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade (padrão: 42)"
    )

    args = parser.parse_args()

    # Validar ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        log_info(f"ERRO: Soma dos ratios deve ser 1.0 (atual: {total_ratio:.2f})")
        exit(1)

    convert_nwpu_to_coco(
        nwpu_root_dir=args.nwpu_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )