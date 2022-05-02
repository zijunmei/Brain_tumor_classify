import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


train_path = "corrupted_subsets/training"
valid_path = "corrupted_subsets/validation"
test_path = "corrupted_subsets/testing"
print(os.getcwd())

train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=train_path, target_size=(224,224), batch_size=64, class_mode='sparse')
valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=valid_path, target_size=(224,224), batch_size=64, shuffle = False,class_mode='sparse')
test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=test_path, target_size=(224,224),  batch_size=64, shuffle=False,class_mode='sparse')
"""
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['glioma', 'meningioma', 'notomor','pituitary'], batch_size=20, class_mode='sparse')
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['glioma', 'meningioma', 'notomor','pituitary'], batch_size=20,shuffle = False, class_mode='sparse')
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['glioma', 'meningioma', 'notomor','pituitary'], batch_size=20, shuffle=False, class_mode='sparse')
"""
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3), filters=16, kernel_size=3, strides=1, padding='same', activation = 'relu', name = "Conv"))
model.add(Conv2D(filters=8, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))

model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))

model.add(Flatten(name = "Flatten"))
model.add(Dense(8,activation = 'relu'))
model.add(Dropout(0.15))
model.add(Dense(4,activation = 'softmax', name = "FC"))
model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=3e-4),
    metrics=['accuracy']
)

monitor = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
#%%
history = model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs= 20 ,
    verbose=1,
	callbacks = [monitor]
)
"""
model.compile(optimizer=Adam(lr=3e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#%%
monitor = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
#%%
history = model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs= 1 ,
    verbose=1,
	callbacks = [monitor]
)
"""
#%%
preds = model.evaluate(test_batches)
print('Test Accracy is '+ str(preds[1]*100)+ '%')
#%%
model.save('CNN_self.h5')

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()





