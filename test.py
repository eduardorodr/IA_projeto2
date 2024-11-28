
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