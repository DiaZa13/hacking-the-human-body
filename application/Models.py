import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import streamlit as st
from PIL import Image
import plotly.graph_objs as go

SEED = 197535
IMG_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 8
AUG_TEST_ARGS = dict(rescale=1. / 255)


@st.experimental_memo
def rle2mask(rle, shape):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


@st.experimental_memo
def get_cells(imgs: [str], organs: [str], src_path: str):
    fig, ax = plt.subplots(1, 5)
    cont = 0
    for img in imgs:
        ax[cont].imshow(io.imread(src_path + img))
        ax[cont].set_title(organs[cont])
        cont += 1
    fig.tight_layout()
    return fig


@st.experimental_memo
def masks_cells(data, src_path: str, imgs: [str], ):
    figure, ax = plt.subplots(1, 3)
    # ax[row].set_title(organs[row])
    img = io.imread(src_path + imgs[4])
    ax[0].imshow(img)
    img_id = int(imgs[4].split('.')[0])
    rle = data.loc[img_id, 'rle']
    mask = rle2mask(rle, img.shape[:2])
    ax[1].imshow(mask, cmap='inferno')
    ax[2].imshow(img)
    ax[2].imshow(mask, cmap='inferno', alpha=0.3)
    return figure


def prediction_preprocess(image):
    img = np.array(image)
    img = img / 255.
    img = np.expand_dims(img, 0)
    return img


def compare(segmentation_model, img, treshold):
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    img = prediction_preprocess(img)
    prediction = segmentation_model.predict(img)
    prediction = (prediction > treshold).astype(np.uint8)
    ax[0].imshow(img[0, :, :, ])
    ax[1].imshow(prediction[0, :, :, ], cmap='inferno')

    return fig


def compare_predictions(segmentation_model, data, loop: int, treshold: float, src_path: str, mask_path: str):
    fig, ax = plt.subplots(loop, 3, figsize=(15, 8))
    for x in range(loop):
        img = Image.open(src_path + data[x])
        img = prediction_preprocess(img)
        mask = Image.open(mask_path + data[x])
        prediction = segmentation_model.predict(img)
        prediction = (prediction > treshold).astype(np.uint8)
        ax[x][0].imshow(img[0, :, :, ])
        ax[x][1].imshow(mask, cmap='inferno')
        ax[x][2].imshow(prediction[0, :, :, ], cmap='inferno')

    return fig


def interactive_history_plot(history, model_name, metrics=['iou_score', 'val_iou_score'], loss=['loss', 'val_loss']):
    train_metric = history[metrics[0]]
    val_metric = history[metrics[1]]
    train_loss = history[loss[0]]
    val_loss = history[loss[1]]

    epochs = range(1, len(train_metric) + 1)

    fig = go.Figure([
        go.Scatter(
            name='train_acc',
            x=np.array(epochs),
            y=train_metric,
            mode='lines',
            marker=dict(color='blue', size=2),
            showlegend=True
        ),
        go.Scatter(
            name='val_acc',
            x=np.array(epochs),
            y=val_metric,
            mode='lines',
            marker=dict(color="blue"),
            line=dict(width=1),
            showlegend=True
        ),
        go.Scatter(
            name='train_loss',
            x=np.array(epochs),
            y=train_loss,
            mode='lines',
            marker=dict(color='orange', size=2),
            showlegend=True
        ),
        go.Scatter(
            name='val_loss',
            x=np.array(epochs),
            y=val_loss,
            mode='lines',
            marker=dict(color="orange"),
            line=dict(width=1),
            showlegend=True
        ),
    ])

    fig.update_layout(
        yaxis_title='Épocas',
        title='Training and validation score and loss in ' + model_name,
        hovermode="x",
        font_family="Balto",
        title_font_family="Times New Roman",
    )

    return fig
