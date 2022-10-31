# Basado en: https://github.com/tdenzl/BuLiAn/blob/main/BuLiAn.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Models import masks_cells, get_cells

# constants
IMG_PATH = 'images/'
MASK_PATH = 'masks/'
plt.rcParams['figure.figsize'] = (15, 15)

# page configuration
st.set_page_config(page_title='HackingHumanBody', page_icon='random', layout='wide')

header = st.container()
dataset = st.container()
models = st.container()

with header:
    st.title('Hacking the Human Body - Finding FTU´s')
    st.caption('Source code [Covid Evolution Analyzer](https://www.linkedin.com/in/tim-denzler/)')
    # description of the analyzer and COVID-19
    st.markdown(
        'El ser humano es en la actualidad uno de los seres vivos multicelulares más complejos, con más de 37 billones de células, las cuales son la unidad'
        'estructural y funcional básica de todo organismo vivo, ya que tienen la capacidad de obtener y utilizar energía, de crecer, morir y autorregularse. '
        'Las células relacionadas a nivel funcional o estructural constituyen lo que se conoce como tejidos, y a su vez, una colección de tejidos especializada '
        'para realizar una función determinada forma una unidad funcional conocida como órgano.'

        'Según Oxford la medicina es la ciencia encargada de estudiar las enfermedades que afectan al ser humano, a lo largo de los años y de la mano de la tecnología '
        'ha logrado encontrar tratamiento y manera de prevenir una variedad de enfermedades, desde el desarrollo de la vacuna contra la polio hasta la identificación '
        'de unidades de tejido funcional en diferentes órganos.'

        'Una unidad de tejido funcional según De Bono se define como una estructura tridimensional de células centradas alrededor de un capilar, en la cual, cada '
        'célula está dentro de la distancia de difusión de cualquier otra célula dentro del mismo bloque, básicamente son estructuras de un tejido que realizan la '
        'función de dicho tejido y suelen replicarse un gran número de veces en un mismo órgano. Las unidades de tejido funcional tienen gran importancia tanto a '
        'nivel médico como patológico, ya que se considera que permiten establecer la relación entre el nivel de escala de todo el cuerpo a nivel micrómetro de '
        'células individuales, es decir, permiten comprender la relación entre las células y la organización de los tejidos.')

    st.image('https://dr282zn36sxxg.cloudfront.net/datastreams/f-d%3Ad1579dcfb8e245073256df525fbb49d37a1f98a648ccb4d8bab50c8c%2BIMAGE_TINY%2BIMAGE_TINY.1')
    st.caption('Fuente: CK-12 Foundation')

with dataset:
    st.header("Recopilación de células de 5 órganos diferentes ")
    # description
    st.markdown(
        'El conjunto de datos para la elaboración de análisis de los tejidos funcionales de los órganos fue obtenido mediante la página de Kaggle de la competencia '
        'Hacking the Human Body. Los datos proporcionados constan de 351 observaciones (con sus respectivas imágenes para el análisis, de las cuales se puede observar '
        'una previsualización en la imagen anterior) y 10 variables, en las cuales se describe tanto información de las imágenes como el sexo y la edad del donador. '
        'Los organos provistos para el análisis son: riñón, pulmón, bazo, intestino grueso y próstata.')

    # read the dataset
    data = pd.read_csv('train.csv', index_col='id')
    kidneys = data[data.organ == 'kidney'].count()['organ']
    lungs = data[data.organ == 'lung'].count()['organ']
    spleen = data[data.organ == 'spleen'].count()['organ']
    prostate = data[data.organ == 'prostate'].count()['organ']

    # some organ images
    imgs = ['62.tiff', '127.tiff', '144.tiff', '203.tiff', '435.tiff']
    organs = ['Kidney', 'Lung', 'Spleen', 'Large Intestine', 'Prostate']

    # cell images
    fig = get_cells(imgs, organs, IMG_PATH)
    st.pyplot(fig)

    '''
    ### Estadísticas generales
    '''
    spacer, row, spacer1, row1, spacer2, row2, spacer3, row3, spacer4 = st.columns(
        (.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
    with row:
        str_games = "🩸 "+ str(kidneys)+ " Riñones"
        st.markdown(str_games)
    with row1:
        str_teams = "🫁 "+ str(lungs)+" Pulmones"
        st.markdown(str_teams)
    with row2:
        str_goals = "🥏 "+str(spleen) +" Bazos"
        st.markdown(str_goals)
    with row3:
        str_shots = "💧 "+ str(prostate)+ " Próstatas"
        st.markdown(str_shots)

    st.markdown("")

    # cells and its respective masks
    figure = masks_cells(data, IMG_PATH, imgs)
    st.pyplot(figure)

with models:
    st.header("Predicción de Unidades de tejido funcional ")
    # description
    st.markdown(
        'A continuación se presentan 3 modelos desarrollados con diversas tecnologías, los cuales proveen una predicción sobre las unidades de tejido funcional'
        'en una célula en formato de imagen la cual puedes subir mediante el siguiente apartado')

    see_data = st.expander('Haz click para observar los resultados de entrenamiento 👇')
    with see_data:
        # poner los valores de accuracy obtenido por cada uno y unas dos imágenes para comparar
        # st.dataframe(data=covid_data.reset_index(drop=True))
        st.markdown('test')
    st.text('')

    uploaded_img = st.file_uploader('Choose a file', type=['png', 'jpg', 'tiff'])

    unet, segment_unet, linknet = st.tabs(['🔎 Classic UNet', '🔬 Segment Models UNet ', '🔭 LinkNet'])

    if uploaded_img is not None:
        # predicciones con modelos
        # unet.line_chart(data=evolution, x=time, y='new_cases')
        # segment_unet.line_chart(data=evolution, x=time, y='new_deaths')
        # linknet.line_chart(data=evolution, x=time, y='new_recovered')
        st.markdown('otro test')
