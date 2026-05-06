"""Visualization module for Plotly graphs (PCA, UMAP)."""

import os
import json
import numpy as np

from core.config.config import get_config, log_info


def add_click_to_open_image(html_path, image_dir='bbox_debug'):
    """
    Add JavaScript to Plotly HTML to open image on click.

    Args:
        html_path: Path to HTML file
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
                closeText.innerHTML = 'Click to close - Image: ' + imageName;
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
    Visualize embeddings saved in JSON file using Plotly Express.

    Args:
        json_path: Path to JSON file with embeddings (default: 'bbox_debug/embeddings.json')
        return_for_notebook: If True, returns figures for notebook display instead of just saving

    Returns:
        If return_for_notebook=True, returns dict with Plotly figures: {'pca_2d': fig, 'pca_3d': fig, 'umap_2d': fig}
    """
    try:
        import plotly.express as px
        import pandas as pd
        import webbrowser
        from sklearn.decomposition import PCA
        import umap
        has_umap = True
    except ImportError as e:
        if 'umap' in str(e):
            has_umap = False
            import plotly.express as px
            import pandas as pd
            import webbrowser
            from sklearn.decomposition import PCA
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

    # Get visualization config
    vis_config = get_config("embedding_visualization") or {}
    pca_config = vis_config.get("pca", {})
    umap_config = vis_config.get("umap", {})

    # PCA 2D
    pca_n_components_2d = pca_config.get("n_components_2d", 2)
    pca_2d = PCA(n_components=pca_n_components_2d)
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

    # PCA 3D
    pca_n_components_3d = pca_config.get("n_components_3d", 3)
    pca_3d = PCA(n_components=pca_n_components_3d)
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

    # UMAP 2D
    fig_umap_2d = None
    if has_umap:
        try:
            umap_n_neighbors = umap_config.get("n_neighbors", 15)
            umap_min_dist = umap_config.get("min_dist", 0.1)
            umap_metric = umap_config.get("metric", "euclidean")
            umap_random_state = umap_config.get("random_state", 42)
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=umap_random_state
            )
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
            'umap_2d': fig_umap_2d
        }
        return results

    log_info("Opening Browser...")
    try:
        webbrowser.open('file://' + os.path.abspath(pca_3d_path))
    except Exception as e:
        log_info(f"Error opening in browser: {e}")
        log_info(f" Open Manually: {pca_3d_path}")