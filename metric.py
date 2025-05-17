import numpy as np
from scipy.stats import pearsonr
import torch

def correlation_dissimilarity(emb1, emb2):
    """
    emb1 (np.array) : embedding in one feature space
    emb2 (np.array) : embedding in another feature space
    """
    dissim1 = 1. - np.corrcoef(emb1)
    dissim2 = 1. - np.corrcoef(emb2)

    triu_indices = np.triu_indices_from(dissim1, k=1)
    flat1 = dissim1[triu_indices]
    flat2 = dissim2[triu_indices]

    # Compute second-order similarity (Pearson correlation)
    r, _ = pearsonr(flat1, flat2)
    return r


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import preprocessing

def train_linear_classifier(X, y, test_size=0.2, random_state=42, **kwargs):
    """
    Trains a linear classifier (Logistic Regression) and returns the model and accuracy.

    Parameters:
    X (array-like): Feature matrix
    y (array-like): Target vector
    test_size (float): Proportion of data to use for testing (default: 0.2)
    random_state (int): Random seed for reproducibility (default: 42)
    **kwargs: Additional arguments to pass to LogisticRegression

    Returns:
    tuple: (trained_model, accuracy_score)
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the linear classifier
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def encode_set(encoder_function: callable, loader, original_loader, device="cpu"):
    all_embeddings = []
    all_labels = []
    all_original_images = []

    with torch.no_grad():
        for (images_dino, targets), (images_orig, _) in zip(loader, original_loader):
            images_dino = images_dino.to(device, non_blocking=True)
            embeddings = encoder_function(images_dino).cpu()
            all_embeddings.append(embeddings)
            all_labels.append(targets)
            all_original_images.append(images_orig)

    embeddings = torch.cat(all_embeddings)  # [N, D]
    labels = torch.cat(all_labels)
    original_images = torch.cat(all_original_images)
    original_images = original_images.reshape(original_images.shape[0], -1)

    return (embeddings.numpy(),
            labels.numpy(),
            original_images.numpy())



from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.palettes import Viridis256
from bokeh.layouts import row

import plotly.express as px

def embedding_plotter(embedding, data=None, hue=None, hover=None, tools = None, nv_cat = 5, height = 400, width = 400, display_result=True):
    '''
    Рисовалка эмбеддинга. 2D renderer: bokeh. 3D renderer: plotly.
    Обязательные инструменты:
        - pan (двигать график)
        - box zoom
        - reset (вылезти из зума в начальное положение)

        embedding: something 2D/3D, slicable ~ embedding[:, 0] - валидно
            Эмбеддинг
        data: pd.DataFrame
            Данные, по которым был построен эмбеддинг
        hue: string
            Колонка из data, по которой красим точки. Поддерживает интерактивную легенду: по клику на каждое
                значение hue можно скрыть весь цвет.
        hover: string or list of strings
            Колонк[а/и] из data, значения которых нужно выводить при наведении мышки на точку
        nv_cat: int
            number of unique values to consider column categorical
        tools: iterable or string in form "tool1,tool2,..." or ["tool1", "tool2", ...]
            tools for the interactive plot
        height, width: int
            parameters of the figure
        display_result: boolean
            if the results are displayed or just returned

    '''
    if tools is None:
        tools = 'lasso_select,box_select,pan,zoom_in,zoom_out,reset,hover'
    else:
        if hover and not("hover" in tools):
            tools = 'hover,'+",".join(tools)


    if embedding.shape[1] == 3:
        if hover:
            hover_data = {h:True for h in hover}
        else:
            hover_data = None
        df = pd.DataFrame(embedding, columns = ['x', 'y', 'z'])
        df = pd.concat((df, data), axis=1)
        fig = px.scatter_3d(
            data_frame = df,
            x='x',
            y='y',
            z='z',
            color=df[hue],
            hover_data = hover_data
        )

        fig.update_layout(
            modebar_add=tools.split(","),
        )

        fig.update_traces(marker_size=1, selector=dict(type='scatter3d'))

        if display_result: fig.show()

    if embedding.shape[1] == 2:
        output_notebook()
        df = pd.DataFrame(embedding, columns = ['x', 'y'])
        df = pd.concat((df, data), axis=1)
        tooltips = [
            ('x, y', '$x, $y'),
            ('index', '$index')
        ]
        if hover:
            for col in hover:
                tooltips.append((col, "@"+col))
        fig = figure(tools=tools, width=width, height=height, tooltips=tooltips)
        if df[hue].nunique() < nv_cat or df[hue].dtype == "category":
            df[hue] = df[hue].astype(str)
            source = ColumnDataSource(df)
            color_mapper = factor_cmap(
            field_name=hue,
            palette='Category10_3',
            factors=df[hue].unique()
            )
            fig.scatter(
            x='x', y='y',
            color=color_mapper,
            source=source,
            legend_group=hue)

            fig.legend.location = 'bottom_left'
            fig.legend.click_policy = 'mute'
        else:
            source = ColumnDataSource(df)
            color_mapper = linear_cmap(
                field_name=hue,
                palette=Viridis256,
                low=min(df[hue]),
                high=max(df[hue]))
            fig.scatter(
                x='x', y='y',
                color=color_mapper,
                source=source)
            color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, location=(0,0), title = hue)
            fig.add_layout(color_bar, 'right')


        if display_result: show(fig)

    if embedding.shape[1] > 3:
        print("wrong species, doooooodes")
    else: return fig

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

def get_tsne(embeddings, labels):
    # Create t-SNE
    tsne = TSNE(n_components=2, random_state=1,
                init='pca', n_iter=5000,
                metric='cosine')

    # Fit and transform your data
    tsne_results = tsne.fit_transform(embeddings[:1000])

    # Prepare the data DataFrame correctly
    data_df = pd.DataFrame({
        'label': np.array(labels[:1000])  # Assuming you have labels
        # Add any other columns you want for hover information
    })

    # Call the plotting function correctly
    embedding_plotter(
        embedding=tsne_results,  # This should be your 2D t-SNE results (1000x2 array)
        data=data_df,            # This contains your labels and other metadata
        hue='label',             # Column name in data_df to use for coloring
    )
    data_df['tsne_x'] = tsne_results[:,0]
    data_df['tsne_y'] = tsne_results[:,1]

    return data_df


def run(encoder_function: callable, loader: torch.utils.data.DataLoader, original_loader: torch.utils.data.DataLoader,
        logger = None, device = "cpu", embeddings_np = None, labels_np = None, original_images_np = None, **kwargs):
    if embeddings_np is None or labels_np is None or original_images_np is None:
        embeddings_np, labels_np, original_images_np = encode_set(encoder_function, loader, original_loader, device)
    cl_acc = train_linear_classifier(embeddings_np, labels_np, **kwargs)[1]
    cor_diss = correlation_dissimilarity(embeddings_np, original_images_np)
    if not logger is None:
        logger.log({
            "classification_accuracy" : cl_acc,
            "second_order_similarity" : cor_diss
        })

    print(f"classification_accuracy : {cl_acc}, \nsecond_order_similarity : {cor_diss}")
    data_df = get_tsne(embeddings_np, labels_np)
    return embeddings_np, labels_np, original_images_np, data_df
