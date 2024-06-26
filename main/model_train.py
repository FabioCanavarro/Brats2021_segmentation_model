import tensorflow.keras as kr
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000 )])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#load data
inputimg = np.load(r'/home/red/COMPDEEP/main/input_dataset.npy')
outputimg = np.load(r'/home/red/COMPDEEP/main/output_dataset.npy')

inputimg = np.array(inputimg, dtype=np.float32)
outputimg = np.array(outputimg, dtype=np.float32)
outputimg = np.expand_dims(outputimg, axis=3)

x_train = inputimg[:1126].copy()
y_train = outputimg[:1126].copy()
x_train.shape[0] == y_train.shape[0]
x_eval = inputimg[1126:1189].copy()
y_eval = outputimg[1126:1189].copy()
print(x_eval.shape, y_eval.shape)
print(x_eval.shape[0] == y_eval.shape[0])
x_test = inputimg[1189:].copy()
y_test = outputimg[1189:].copy()

print("train target:",y_train.shape,"                test target:",y_test.shape)
print(f"train x{x_test.shape}","                test x:", x_test.shape)

#shuffle data
np.random.seed(0)
np.random.shuffle(x_train)
np.random.shuffle(y_train)
np.random.shuffle(x_eval)
np.random.shuffle(y_eval)

np.random.shuffle(x_test)
np.random.shuffle(y_test)

y_train = y_train.astype(int)
y_eval = y_eval.astype(int)
y_test = y_test.astype(int)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

input_shape = (240, 240, 4)
num_classes = 4

# Data preparation functions
#why my last code didnt work:
#the labels in the output goes 0,1,2,4????????
#yeah 0,1,2,4 not 0,1,2,3 but 0,1,2,4
#no wonder i am now depressed

#side note: np.unique(tells us all unique element)
def correct_labels(y):
    y[y == 4] = 3
    return y
#normalize data
def normalize_data(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Apply corrections to your data
x_train = normalize_data(x_train)
x_eval = normalize_data(x_eval)
x_test = normalize_data(x_test)

y_train = correct_labels(y_train)
y_eval = correct_labels(y_eval)
y_test = correct_labels(y_test)

# Rest of your code remains the same
def unet_model(input_shape, num_classes):
    inputs = keras.Input(input_shape)
    
    # Encoder (downsampling)
    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = conv_block(pool3, 512)
    
    # Decoder (upsampling)
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.concatenate([up5, conv3])
    conv5 = conv_block(up5, 256)
    
    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.concatenate([up6, conv2])
    conv6 = conv_block(up6, 128)
    
    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.concatenate([up7, conv1])
    conv7 = conv_block(up7, 64)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv7)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

model = unet_model(input_shape, num_classes)

optimizer = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Print model summary
model.summary()

# Ensure your data is properly shaped
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

import tensorflow as tf

class_weights = {0: 1.0, 1: 2.0, 2: 2.0, 3: 2.0}  # Adjust these based on class frequencies

def weighted_sparse_categorical_crossentropy(y_true, y_pred):
    weights = tf.gather(list(class_weights.values()), tf.cast(y_true, tf.int32))
    unweighted_losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    weighted_losses = unweighted_losses * weights
    return tf.reduce_mean(weighted_losses)
y_train = np.squeeze(y_train, axis=-1)
y_eval = np.squeeze(y_eval, axis=-1)
y_test = np.squeeze(y_test, axis=-1)
model.compile(optimizer='adam', loss=weighted_sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(
    x_train, y_train,
    validation_data=(x_eval, y_eval),
    batch_size=16,
    epochs=80,
    callbacks=[
        keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5),
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)
with open('model_history.txt', 'w') as f:
    f.write(str(history.history))
    
