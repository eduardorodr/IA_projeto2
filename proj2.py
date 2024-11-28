import streamlit as st
import pandas as pd
import numpy as np
import keras
import kagglehub
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = kagglehub.dataset_download("kritikseth/fruit-and-vegetable-image-recognition")
val_path = path+"/validation"
train_path = path+"/train"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_path, seed=2509, image_size=(224, 224), batch_size=32)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(val_path, seed=2509, image_size=(224, 224), shuffle=False, batch_size=32)
class_names = train_dataset.class_names
model = keras.saving.load_model("modelo.keras")

# Tituo do projeto e sidebar
st.title("Classificador de Frutas - CNN")
st.sidebar.title("Menu")
selecao = st.sidebar.radio("Navegue pelo projeto:", ["Descrição do Projeto", "Classificação de Frutas", "Desempenho","Conclusão"])
##

# Seção de Descrição do Projeto
if selecao == "Descrição do Projeto":
    st.header("Descrição do Projeto")

    # Introdução 
    st.subheader("Introdução")
    st.write('''

            Este trabalho foi desenvolvido como segunda avaliação da disciplina de
            Inteligência Artificial, para solidificar, exercitar e exemplificar os
            conhecimentos que foram obtidos ao longo do semestre, principalmente os
            que tocam em classificação, modelos de redes neurais, interpretação de
            datasets, linguagem python e suas bibliotecas e publicação de aplicações.
            O tema deste projeto, a Classificação de Frutas utilizando imagens, foi
            escolhido tendo em vista a proximidade com temas e aplicações dados de
            exemplo durante as aulas e a grande disponibilidade de material, tanto em
            dados (imagens de frutas) quanto documentação do uso de imagens de frutas
            em modelos de classificação.\n
            O modelo escolhido, CNN (Redes Neurais Convolucionais) é
            amplamente utilizado na classificação com imagens, é extremamente
            documentado e explorado nas comunidades de desenvolvedores de machine
            learning. Utilizando bibliotecas que possuem o modelo já implementado, é
            possível adaptá-lo para classificar frutas com facilidade e em poucas etapas.\n
            O dataset escolhido é pouco representativo pelo seu pequeno número
            de amostras, mas os resultados apresentados foram satisfatórios, dado que a
            escolha do conjunto de dados foi condicionada às limitações de hardware do
            computador onde o modelo foi treinado. Posteriormente neste relatório serão
            abordados os problemas que foram encontrados durante o desenvolvimento
            relacionados a hardware.

    ''')

    img_banner = Image.open("frutas.jpg")
    img_banner = img_banner.resize([1000,400],3)
    st.image(img_banner)

##

# Seção de Classificação de Frutas // Código para treino e teste do modelo, envio de arquivos para classificação de uma amostra
elif selecao == "Classificação de Frutas":
    st.header("Classificação de Frutas")

    # Explicação geral do código - test.py, salvar modelo e iniciar modelo salvo
    st.subheader("Código do projeto")
    st.write('O código do projeto foi desenvolvido e testado no notebook da plataforma Kaggle e posteriormente localmente')

    code = '''
        
        import kagglehub
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
        from tensorflow.keras.layers import BatchNormalization

        path = kagglehub.dataset_download("kritikseth/fruit-and-vegetable-image-recognition")
        val_path= path+"/validation"
        train_path= path+"/train"

        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_path, seed=2509, image_size=(224, 224), batch_size=32)
        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(val_path, seed=2509, image_size=(224, 224), shuffle=False, batch_size=32)
        class_names = train_dataset.class_names

        model = Sequential()
        model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(224,224,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(len(class_names),activation='softmax'))

        model.compile( loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = Adam(learning_rate=0.001), metrics = ["accuracy"])

        history = model.fit(x=train_dataset, epochs= 20, validation_data=val_dataset)
        model.save("modelo.keras")
    
    '''

    st.write(''' Download do modelo ''')
    img_banner = Image.open("modelo.PNG")
    st.image(img_banner)

    st.write(''' Treino ''')
    img_banner = Image.open("treino.PNG")
    st.image(img_banner)

    st.code(code, language="python")

    amostra = st.file_uploader("Amostra de Fruta", type=None, accept_multiple_files=False, key=None, 
        help="Faça um upload de uma imagem que contenha uma fruta para que o modelo possa adivinhar!", 
        on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    #

    if amostra is not None:

        imgamostra = Image.open(amostra)
        image_array = np.array(imgamostra)
        fig, ax = plt.subplots()
        ax.imshow(image_array)
        ax.axis('off')  
        st.pyplot(fig)

        img = imgamostra.resize([224,224],3)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model.predict(images, batch_size=32)
        st.write("Predição: %s" % class_names[np.argmax(pred)] )
    #
    
##

# Seção de Desempenho // exibir graficos de acurácia e perda do modelo
elif selecao == "Desempenho":
    
    st.header("Desempenho")

    img_banner = Image.open("acu.PNG")
    st.image(img_banner)

    img_banner = Image.open("loss.PNG")
    st.image(img_banner)

    st.write('''

        O desempenho do modelo foi analisado sob diferentes configurações de arquitetura: diferentes tamanhos de batch, número de épocas,
        taxas de aprendizado e número de camadas. A maior acurácia foi 0.95, embora na comunidade do Kaggle outros usuários tenham alcançado 
        cerca de 0.98 de acurácia ainda usando o CNN.
             
    ''')

##
# Seção de Conclusão // comentar a respeito das dificuldades, resultados esperados e obtidos
else:
    
    st.header("Conclusão")
    st.subheader("Pinneaple??")

    img_banner = Image.open("abacaxi.PNG")
    st.image(img_banner)

    st.write('''

        Uma amostra pouco representativa pode gerar resultados inesperados!
             
    ''')

##