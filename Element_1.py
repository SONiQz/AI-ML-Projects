from datetime import datetime
import os
import tensorflow as tf
from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

model_name = 'Test'

log_dir = "logs/fit/" + model_name + " Test_1 - " + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                     )

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

input_shape = (256, 256, 3)
target_size = (256, 256)
batch_size = 32
epochs = 40

model_name = 'Test'

model_filename2 = (model_name + '1' + '.hdf5')

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=5,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=5,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'Element 1/train',
    target_size=target_size,
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical',
    interpolation='nearest')

validation_generator = validation_datagen.flow_from_directory(
    'Element 1/validation',
    target_size=target_size,
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical',
    interpolation='nearest')

optimizer = Adam(learning_rate=0.0001)
loss = ['categorical_crossentropy']
metrics = ['CategoricalAccuracy']

model = Sequential([
    layers.Conv2D(64, (6, 6), input_shape=input_shape),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, (6, 6)),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(256, (6, 6)),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(512, (6, 6)),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),

    layers.Dense(256),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(10),
    layers.Activation('softmax'),

])

model.summary()

model.compile(optimizer, loss, metrics)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filename2, monitor='CategoricalAccuracy',)


build_model = model.fit(
    train_generator,
    steps_per_epoch=3054 // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=194 // batch_size,
    callbacks=[model_checkpoint, tensorboard_callback],
    use_multiprocessing=False,
    verbose=1, )

