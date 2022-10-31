import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import streamlit as st


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
