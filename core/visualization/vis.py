"""Visualization module for model outputs, heatmaps, and bounding boxes."""
import matplotlib
matplotlib.use('Agg')

import os
import cv2
import json
import torch
import webbrowser
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from core.config.config import get_config, log_info
from ..datasets import get_data_loaders
from .vis_utils import VisUtils, compute_dynamic_threshold
from ..metrics import calculate_metric, calculate_CorLoc, calculate_map
from .gradcam import compute_gradcam, overlay_heatmap
from .attention_rollout import compute_attention_rollout


def get_root_path():
    """Get the data path of the active dataset from configuration."""
    return get_config("datasets", get_config("active_dataset"), "data_path")

ROOT_PATH = get_root_path()

def embedding_vis(model, device, model_name, return_for_notebook=False):
    """
    Visualize embeddings distribution using heatmaps and boxplots.

    Generates embedding representations for fire and non-fire samples,
    computes distances, and creates visualizations showing clustering patterns.

    Args:
        model: The trained model with backbone attribute
        device: Device to run the model on (CPU/GPU)
        model_name: Name of the model for configuration lookup
        return_for_notebook: If True, returns figures and data for notebook display instead of saving files

    Returns:
        If return_for_notebook=True, returns dict with:
        {
            'train': {'heatmap_fig': Figure, 'boxplot_fig': Figure, 'data': dict},
            'test': {'heatmap_fig': Figure, 'boxplot_fig': Figure, 'data': dict}
        }
    """
    img_size = 224
    if get_config("models", model_name, "img_size") is not None:
        img_size = int(get_config("models", model_name, "img_size"))

    data_loaders, datasets = get_data_loaders(ROOT_PATH, batch_size=8, num_workers=0, img_size=img_size)

    def create_emb(model, device, model_name, type_dataloader):
        """Generate and visualize embeddings for fire and non-fire samples."""
        embeddings_fire = []
        embeddings_not = []
        labels_fire = []
        labels_not = []

        model.eval()

        for idx, (features, labels, paths) in enumerate(data_loaders[type_dataloader]):
            features, labels = features.to(device), labels.to(device)

            with torch.no_grad():
                _, embeddings_batch = model(features, return_embedding=True)

                for embedding, label, path in zip(embeddings_batch, labels, paths):
                    if label == 0:
                        embeddings_fire.append(embedding.cpu())
                        labels_fire.append(os.path.basename(path))
                    else:
                        embeddings_not.append(embedding.cpu())
                        labels_not.append(os.path.basename(path))

        embeddings_fire = torch.stack(embeddings_fire)
        embeddings_not = torch.stack(embeddings_not)
        dist_matrix = torch.cdist(embeddings_fire, embeddings_not)

        distances_output = []
        for i, label_not in enumerate(labels_not):
            for j, label_fire in enumerate(labels_fire):
                dist = dist_matrix[j][i].item()
                if dist < 40:
                    distances_output.append({'fire_img': label_fire, 'not_img': label_not, 'distance': dist})

        if return_for_notebook:
            import io
            from PIL import Image
            import matplotlib

            original_backend = matplotlib.get_backend()
            matplotlib.use('Agg')
            sns.set(font_scale=0.2)

            fig_heatmap, ax_heatmap = plt.subplots(figsize=(20, 20))
            sns.heatmap(dist_matrix.cpu().numpy(), xticklabels=labels_not,
                        yticklabels=labels_fire, annot=False, cmap="coolwarm", ax=ax_heatmap)
            ax_heatmap.set_title(f'Heatmap de Distâncias - {type_dataloader}')
            fig_heatmap.tight_layout()

            buf_heatmap = io.BytesIO()
            fig_heatmap.savefig(buf_heatmap, format='png', dpi=100, bbox_inches='tight')
            buf_heatmap.seek(0)
            img_heatmap = Image.open(buf_heatmap)
            plt.close(fig_heatmap)

            # Create boxplot figure
            fig_boxplot, ax_boxplot = plt.subplots(figsize=(len(labels_not) * 0.3, 10))
            ax_boxplot.boxplot(dist_matrix.cpu().numpy(), vert=True)
            ax_boxplot.set_xticks(range(1, len(labels_not) + 1))
            ax_boxplot.set_xticklabels(labels_not, rotation=90, fontsize=10)
            ax_boxplot.tick_params(axis='y', labelsize=14)
            ax_boxplot.set_xlabel('Imagens Not Fire')
            ax_boxplot.set_ylabel('Distância')
            ax_boxplot.set_title(f'Boxplot de Distâncias - {type_dataloader}')
            ax_boxplot.grid(True)
            fig_boxplot.tight_layout()

            # Convert boxplot to image
            buf_boxplot = io.BytesIO()
            fig_boxplot.savefig(buf_boxplot, format='png', dpi=100, bbox_inches='tight')
            buf_boxplot.seek(0)
            img_boxplot = Image.open(buf_boxplot)
            plt.close(fig_boxplot)

            # Restore original backend
            matplotlib.use(original_backend)

            return {
                'heatmap_fig': img_heatmap,
                'boxplot_fig': img_boxplot,
                'data': {
                    'embeddings_fire': embeddings_fire.cpu().numpy(),
                    'embeddings_not': embeddings_not.cpu().numpy(),
                    'labels_fire': labels_fire,
                    'labels_not': labels_not,
                    'dist_matrix': dist_matrix.cpu().numpy(),
                    'close_pairs': distances_output
                }
            }
        else:
            # Original behavior: save to files
            plt.ioff()
            sns.set(font_scale=0.2)
            plt.figure(figsize=(200, 200))
            sns.heatmap(dist_matrix.cpu().numpy(), xticklabels=labels_not,
                        yticklabels=labels_fire, annot=False, cmap="coolwarm")
            plt.savefig(f'heatmap_{type_dataloader}.png', bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=(len(labels_not) * 0.3, 20))
            plt.boxplot(dist_matrix.cpu().numpy(), vert=True)
            plt.xticks(ticks=range(1, len(labels_not) + 1), labels=labels_not, rotation=90, fontsize=10)
            plt.yticks(fontsize=14)
            plt.xlabel('Column index (embeddings_not)')
            plt.ylabel('Distance')
            plt.title('Boxplot of distances by column (embeddings_not)')
            plt.grid(True)
            plt.savefig(f'boxplot_{type_dataloader}.png', bbox_inches='tight')
            plt.close()

            distances_text = [f'{d["not_img"]},{d["fire_img"]} -> {d["distance"]:.4f}' for d in distances_output]
            with open(f"resultados_distancias_{type_dataloader}.txt", "w") as f:
                for line in distances_text:
                    f.write(line + "\n")

            return None

    if return_for_notebook:
        # Close all previous figures before creating new ones
        plt.close('all')

        results = {
            'train': create_emb(model, device, model_name, "Train"),
            'test': create_emb(model, device, model_name, "Test")
        }
        return results
    else:
        create_emb(model, device, model_name, "Train")
        create_emb(model, device, model_name, "Test")
        return None

def vis(model, device, model_name, save_file=False):
    """
    Visualize model predictions on test images.

    Generates heatmaps, bounding boxes, and overlays for fire detection.
    Optionally saves visualizations to disk.

    Args:
        model: The trained model
        device: Device to run the model on (CPU/GPU)
        model_name: Name of the model for configuration lookup
        save_file: If True, save visualizations to results directory
    """
    if save_file:
        log_info("Starting visualization saving process")
        os.makedirs(f'{os.getcwd()}/results', exist_ok=True)

    model.eval()
    img_size = 224
    if get_config("models", model_name, "img_size") is not None:
        img_size = int(get_config("models", model_name, "img_size"))

    data_loaders, datasets = get_data_loaders(ROOT_PATH, batch_size=8, num_workers=0, img_size=img_size)

    for idx, (features, labels, paths) in enumerate(data_loaders["Test"]):
        features, labels = features.to(device), labels.to(device)
        for feature, label, path in zip(features, labels, paths):
            if label == 0:
                img_np, plt_img, metric = vis_one(model, model_name, feature.unsqueeze(0), path)
                VisUtils.plot_image(img_np, plt_img, metric, save_file, path)

    if save_file:
        log_info("Visualization saving process completed")

def vis_one(model, model_name, img_tensor, img_path):
    """
    Generate visualization for a single image with heatmap and bounding boxes.

    Computes attention/gradient-based heatmaps and generates bounding boxes
    from the heatmap. Overlays ground truth boxes if available.

    Args:
        model: The trained model
        model_name: Name of the model for configuration lookup
        img_tensor: Image tensor with shape (1, 3, H, W)
        img_path: Path to the image file

    Returns:
        img_np: Denormalized image with drawn boxes
        plt_img: Heatmap overlay visualization
        metric: Accuracy metric comparing predictions to ground truth
    """
    bbox_to_id, filename_to_id, image_sizes = VisUtils.get_image_ids()
    img_id = filename_to_id.get(os.path.basename(img_path))
    if "Dino" in model_name or "DeiT" in model_name or "ViT" in model_name or "DaViT" in model_name or "Swin" in model_name:
        rollout_config = get_config("attention_rollout") or {}
        heatmap = compute_attention_rollout(
            model, model_name, img_tensor
        )
    else:
        heatmap = compute_gradcam(model, model_name, img_tensor)

    heatmap = cv2.resize(heatmap, (img_tensor.shape[2], img_tensor.shape[3]))

    min_area = get_config("defaults", "min_bbox_area")
    if min_area is None:
        min_area = 1600

    # Use dynamic threshold (Otsu) instead of fixed threshold
    gradcam_threshold = compute_dynamic_threshold(heatmap, method="otsu", fire_adapted=True)
    bbs = VisUtils.generate_bounding_box(heatmap, threshold=gradcam_threshold, min_area=min_area)
    plt_img, _ = overlay_heatmap(img_path, heatmap)

    plt_bbs = img_tensor.squeeze(0).detach().clone()
    for bb in bbs:
        plt_bbs = VisUtils.draw_rectangle(plt_bbs, bb[:4])

    if img_id:
        for bb in np.array(bbox_to_id.get(img_id)):
            if img_id in image_sizes:
                original_width, original_height = image_sizes[img_id]
                or_size = min(original_width, original_height)
                x, y, width, height = VisUtils.redimension_bboxes(
                    (int(round(v)) for v in bb[:4]),
                    plt_bbs.shape[2],
                    or_size,
                    original_width,
                    original_height
                )
            else:
                or_size = 256
                x, y, width, height = VisUtils.redimension_bboxes((int(round(v)) for v in bb[:4]), plt_bbs.shape[2], or_size)
            plt_bbs = VisUtils.draw_rectangle(plt_bbs, (x, y, x + width, y + height), (0, 1, 0))

    metric  = calculate_metric([], bbox_to_id.get(img_id, []), bbs, 0.5, plt_bbs.shape[2], img_id, image_sizes)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = plt_bbs.cpu().permute(1, 2, 0).numpy() * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    img_np = np.clip(img_np, 0, 1)

    return img_np, plt_img, metric


def evaluate_model(model, model_name, device, data_loaders, bbox_to_id, filename_to_id, image_sizes, threshold=0.5, metric='box_acc'):
    """
    Evaluate model using dynamic Otsu threshold per image.

    Args:
        model: Model to evaluate
        model_name: Model name
        device: Device (CPU/GPU)
        data_loaders: Dict with data loaders
        bbox_to_id: Mapping of image_id to bounding boxes
        filename_to_id: Mapping of filename to image_id
        image_sizes: Dict mapping image_id to original (width, height)
        threshold: IoU/LocAcc threshold (default: 0.5)
        metric: 'box_acc', 'mAP', or 'CorLoc' (default: 'box_acc')

    Returns:
        result: Metric value
    """
    total_acc = []
    all_predictions = []

    log_info(f"Evaluating with Otsu dynamic threshold per image")

    for idx, (features, labels, paths) in enumerate(data_loaders["Test"]):
        features, labels = features.to(device), labels.to(device)

        for feature, label, path in zip(features, labels, paths):
            from core.config.config import get_active_dataset_config
            num_classes = get_active_dataset_config('num_classes')

            if num_classes == 2 and label != 0:
                continue

            feature_unsqueeze = feature.unsqueeze(0)

            if "Dino" in model_name or "DeiT" in model_name or "ViT" in model_name or "Swin" in model_name:
                heatmap = compute_attention_rollout(
                    model, model_name, feature_unsqueeze
                )
            else:
                heatmap = compute_gradcam(model, model_name, feature_unsqueeze)

            heatmap = cv2.resize(heatmap, (feature_unsqueeze.shape[2], feature_unsqueeze.shape[3]))

            gradcam_threshold = compute_dynamic_threshold(heatmap, method="otsu", fire_adapted=True)
            bbs = VisUtils.generate_bounding_box(heatmap, threshold=gradcam_threshold)

            bbs = VisUtils.postprocess_BB(
                bbs,
                min_confidence=0.3,
                remove_nested=True,
                containment_threshold=0.8
            )

            img_id = filename_to_id.get(os.path.basename(path))

            if not img_id:
                continue

            if metric == 'box_acc':
                total_acc = calculate_metric(total_acc, bbox_to_id.get(img_id, []), bbs, threshold, feature_unsqueeze.shape[3], img_id, image_sizes)
            elif metric in ['mAP', 'CorLoc']:
                all_predictions.append((img_id, bbs, feature_unsqueeze.shape[3], image_sizes, label.item()))

    if metric == 'box_acc':
        result = sum(total_acc) / len(total_acc) if len(total_acc) > 0 else 0.0
    elif metric == 'mAP':
        result = calculate_map(all_predictions, bbox_to_id, filename_to_id)
    elif metric == 'CorLoc':
        result = calculate_CorLoc(all_predictions, bbox_to_id, filename_to_id, loc_threshold=threshold)
    else:
        raise ValueError(f"Metric '{metric}' not recognized. Use 'box_acc', 'mAP', or 'CorLoc'.")

    log_info(f"{metric}: {result:.4f} (Otsu threshold)")
    return result


def test_model(model, model_name, device, num_workers, threshold=0.5, metric='box_acc'):

    img_size = 224
    if get_config("models", model_name, "img_size") is not None:
        img_size = int(get_config("models", model_name, "img_size"))

    data_loaders, datasets = get_data_loaders(ROOT_PATH, batch_size=8, num_workers=num_workers, img_size=img_size)

    bbox_to_id, filename_to_id, image_sizes = VisUtils.get_image_ids()
    return evaluate_model(model, model_name, device, data_loaders, bbox_to_id, filename_to_id, image_sizes, threshold, metric)


def visualize_cropped_embeddings(embeddings_data, output_dir='embedding_visualizations', save_numpy=True):
    """
    Visualize cropped region embeddings in multiple subplots using different reduction techniques.

    Args:
        embeddings_data: Dict with the following keys:
            - 'embeddings': array (N, embedding_dim)
            - 'confidences': array (N,)
            - 'classes': array (N,)
            - 'class_names': dict mapping class_id to name
        output_dir: Directory to save the visualizations
        save_numpy: If True, save embeddings to .npz file
    """
    if embeddings_data is None or len(embeddings_data['embeddings']) == 0:
        log_info("No embeddings available for visualization")
        return

    os.makedirs(output_dir, exist_ok=True)

    embeddings = embeddings_data['embeddings']
    confidences = embeddings_data['confidences']
    classes = embeddings_data['classes']
    class_names = embeddings_data['class_names']

    if save_numpy:
        save_path = os.path.join(output_dir, 'cropped_embeddings.npz')
        np.savez(
            save_path,
            embeddings=embeddings,
            confidences=confidences,
            classes=classes,
            class_names=str(class_names)
        )

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    if has_sklearn:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes = np.array([[axes]])

    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    color_map = {class_id: colors[class_id] for class_id in class_names.keys()}

    if has_sklearn:
        pca_2d = PCA(n_components=2)
        embeddings_pca_2d = pca_2d.fit_transform(embeddings)

        for class_id, class_name in class_names.items():
            mask = classes == class_id
            axes[0, 0].scatter(
                embeddings_pca_2d[mask, 0],
                embeddings_pca_2d[mask, 1],
                c=[color_map[class_id]],
                label=class_name,
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=1
            )

        axes[0, 0].set_title(
            f'PCA 2D ({pca_2d.explained_variance_ratio_.sum():.1%})',
            fontsize=12,
            fontweight='bold'
        )
        axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        pca_3d = PCA(n_components=3)
        embeddings_pca_3d = pca_3d.fit_transform(embeddings)

        for class_id, class_name in class_names.items():
            mask = classes == class_id
            sizes = 50 + confidences[mask] * 150
            axes[0, 1].scatter(
                embeddings_pca_3d[mask, 0],
                embeddings_pca_3d[mask, 1],
                c=[color_map[class_id]],
                label=class_name,
                s=sizes,
                alpha=0.6,
                edgecolors='black',
                linewidth=1
            )

        axes[0, 1].set_title(
            f'PCA 3D → 2D ({pca_3d.explained_variance_ratio_.sum():.1%})',
            fontsize=12,
            fontweight='bold'
        )
        axes[0, 1].set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
        axes[0, 1].set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        try:
            tsne_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_tsne_2d = tsne_2d.fit_transform(embeddings)

            for class_id, class_name in class_names.items():
                mask = classes == class_id
                axes[1, 0].scatter(
                    embeddings_tsne_2d[mask, 0],
                    embeddings_tsne_2d[mask, 1],
                    c=[color_map[class_id]],
                    label=class_name,
                    s=100,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=1
                )

            axes[1, 0].set_title('t-SNE 2D', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('t-SNE 1')
            axes[1, 0].set_ylabel('t-SNE 2')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, 'Error generating t-SNE', ha='center', va='center')

        for class_id, class_name in class_names.items():
            mask = classes == class_id
            axes[1, 1].hist(
                confidences[mask],
                bins=10,
                alpha=0.6,
                label=class_name,
                color=color_map[class_id],
                edgecolor='black'
            )

        axes[1, 1].set_title('Confidence Distribution by Class', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'embedding_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    for class_id, class_name in class_names.items():
        mask = classes == class_id
        count = mask.sum()
        mean_conf = confidences[mask].mean()
        std_conf = confidences[mask].std()
        log_info(f"{class_name}: {count} samples")
        log_info(f"  - Mean confidence: {mean_conf:.4f} ± {std_conf:.4f}")

def add_click_to_open_image(html_path, image_dir='bbox_debug'):
    """
    Add JavaScript to Plotly HTML to open image on click.

    Args:
        html_path: Path to the HTML file
        image_dir: Directory where images are located
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    click_script = f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        var plot = document.getElementsByClassName('plotly-graph-div')[0];

        plot.on('plotly_click', function(data) {{
            var point = data.points[0];
            var imageName = point.customdata;

            if (imageName) {{
                var imagePath = '{image_dir}/' + imageName;

                var modal = document.createElement('div');
                modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.9);z-index:10000;display:flex;justify-content:center;align-items:center;cursor:pointer;';

                var img = document.createElement('img');
                img.src = imagePath;
                img.style.cssText = 'max-width:90%;max-height:90%;border:3px solid white;box-shadow:0 0 20px rgba(255,255,255,0.3);';

                var closeText = document.createElement('div');
                closeText.innerHTML = 'Clique para fechar - Imagem: ' + imageName;
                closeText.style.cssText = 'position:absolute;top:20px;color:white;font-size:18px;font-weight:bold;text-shadow:2px 2px 4px rgba(0,0,0,0.8);';

                modal.appendChild(closeText);
                modal.appendChild(img);
                document.body.appendChild(modal);

                modal.onclick = function() {{
                    document.body.removeChild(modal);
                }};

                img.onerror = function() {{
                    closeText.innerHTML = 'Image not found: ' + imagePath + '<br>Click to close';
                    closeText.style.color = '#ff4444';
                }};
            }}
        }});
    }});
    </script>
    """

    html_content = html_content.replace('</body>', click_script + '</body>')

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def visualize_embeddings_from_json(json_path='bbox_debug/embeddings.json', return_for_notebook=False):
    """
    Visualiza embeddings salvos em arquivo JSON usando Plotly Express.

    Args:
        json_path: Path to the JSON file with embeddings (default: 'bbox_debug/embeddings.json')
        return_for_notebook: If True, returns figures for notebook display instead of only saving

    Returns:
        Se return_for_notebook=True, retorna dict com figuras Plotly: {'pca_2d': fig, 'pca_3d': fig, 'tsne_2d': fig, 'umap_2d': fig}
    """
    try:
        import plotly.express as px
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import umap
        has_umap = True
    except ImportError as e:
        if 'umap' in str(e):
            has_umap = False
            import plotly.express as px
            import pandas as pd
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
        else:
            return

    if not os.path.exists(json_path):
        log_info(f"File Not Found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if len(data) == 0:
        log_info("No embedding found in JSON")
        return

    log_info(f"Total Samples: {len(data)}")
    log_info(f"{'='*60}\n")

    embeddings = np.array([item['embedding'] for item in data])
    classes = np.array([item.get('true_class', item.get('class', 0)) for item in data])
    image_names = [item['image_name'] for item in data]

    class_names = {0: 'Fire', 1: 'No Fire'}
    class_labels = [class_names[c] for c in classes]

    log_info(f"Dimension: {embeddings.shape}")
    log_info(f"Classes: {np.unique(classes)}")

    output_dir = os.path.dirname(json_path)
    if not output_dir:
        output_dir = '.'

    active_dataset = get_config("active_dataset")
    data_path = get_config("datasets", active_dataset, "data_path")
    if not data_path:
        data_path = "data_fire"

    image_dir = os.path.abspath(os.path.join(data_path, "Test", "images"))

    pca_2d = PCA(n_components=2)
    embeddings_pca_2d = pca_2d.fit_transform(embeddings)

    df_pca_2d = pd.DataFrame({
        'PC1': embeddings_pca_2d[:, 0],
        'PC2': embeddings_pca_2d[:, 1],
        'Class': class_labels,
        'Image': image_names
    })

    fig_pca_2d = px.scatter(
        df_pca_2d,
        x='PC1',
        y='PC2',
        color='Class',
        hover_data=['Image'],
        custom_data=['Image'],
        title=f'PCA 2D ({pca_2d.explained_variance_ratio_.sum():.1%})',
        labels={
            'PC1': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
            'PC2': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})'
        },
        color_discrete_map={'Fire': '#FF4B4B', 'No Fire': '#4B9BFF'}
    )

    pca_2d_path = os.path.join(output_dir, 'embeddings_pca_2d.html')
    fig_pca_2d.write_html(pca_2d_path)
    add_click_to_open_image(pca_2d_path, image_dir)

    pca_3d = PCA(n_components=3)
    embeddings_pca_3d = pca_3d.fit_transform(embeddings)

    df_pca_3d = pd.DataFrame({
        'PC1': embeddings_pca_3d[:, 0],
        'PC2': embeddings_pca_3d[:, 1],
        'PC3': embeddings_pca_3d[:, 2],
        'Class': class_labels,
        'Image': image_names
    })

    fig_pca_3d = px.scatter_3d(
        df_pca_3d,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Class',
        hover_data=['Image'],
        custom_data=['Image'],
        title=f'PCA 3D ({pca_3d.explained_variance_ratio_.sum():.1%})',
        labels={
            'PC1': f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
            'PC2': f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
            'PC3': f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})'
        },
        color_discrete_map={'Fire': '#FF4B4B', 'No Fire': '#4B9BFF'}
    )

    pca_3d_path = os.path.join(output_dir, 'embeddings_pca_3d.html')
    fig_pca_3d.write_html(pca_3d_path)
    add_click_to_open_image(pca_3d_path, image_dir)

    try:
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_tsne_2d = tsne_2d.fit_transform(embeddings)

        df_tsne_2d = pd.DataFrame({
            't-SNE 1': embeddings_tsne_2d[:, 0],
            't-SNE 2': embeddings_tsne_2d[:, 1],
            'Class': class_labels,
            'Image': image_names
        })

        fig_tsne_2d = px.scatter(
            df_tsne_2d,
            x='t-SNE 1',
            y='t-SNE 2',
            color='Class',
            hover_data=['Image'],
            custom_data=['Image'],
            title='t-SNE 2D',
            color_discrete_map={'Fire': '#FF4B4B', 'No Fire': '#4B9BFF'}
        )

        tsne_2d_path = os.path.join(output_dir, 'embeddings_tsne_2d.html')
        fig_tsne_2d.write_html(tsne_2d_path)
        add_click_to_open_image(tsne_2d_path, image_dir)
    except Exception as e:
        log_info(f"Error Generating t-SNE: {e}")

    if has_umap:
        try:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_umap_2d = reducer.fit_transform(embeddings)

            df_umap_2d = pd.DataFrame({
                'UMAP 1': embeddings_umap_2d[:, 0],
                'UMAP 2': embeddings_umap_2d[:, 1],
                'Class': class_labels,
                'Image': image_names
            })

            fig_umap_2d = px.scatter(
                df_umap_2d,
                x='UMAP 1',
                y='UMAP 2',
                color='Class',
                hover_data=['Image'],
                custom_data=['Image'],
                title='UMAP 2D',
                color_discrete_map={'Fire': '#FF4B4B', 'No Fire': '#4B9BFF'}
            )

            umap_2d_path = os.path.join(output_dir, 'embeddings_umap_2d.html')
            fig_umap_2d.write_html(umap_2d_path)
            add_click_to_open_image(umap_2d_path, image_dir)
        except Exception as e:
            log_info(f"Error Generating UMAP: {e}")

    for class_id, class_name in class_names.items():
        count = (classes == class_id).sum()
        percentage = (count / len(classes)) * 100

    if return_for_notebook:
        results = {
            'pca_2d': fig_pca_2d,
            'pca_3d': fig_pca_3d,
            'tsne_2d': fig_tsne_2d if 'fig_tsne_2d' in locals() else None,
            'umap_2d': fig_umap_2d if 'fig_umap_2d' in locals() else None
        }
        return results

    log_info("Opening Browser...")
    try:
        webbrowser.open('file://' + os.path.abspath(pca_3d_path))
    except Exception as e:
        log_info(f"Error opening in browser: {e}")
        log_info(f" Open Manually: {pca_3d_path}")


if __name__ == "__main__":
    from ..training.engine import get_device
    from ..models import get_model

    device = get_device()
    model = get_model("Swim_Small", num_classes=2, device=device)
