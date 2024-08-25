import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from PIL import ImageFile
import matplotlib.pyplot as plt
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_size = 64
batch_size = 64

train_dir = 'cartoon/train'
test_dir = 'cartoon/test'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)
classes = ('donald', 'mickey', 'minion', 'olaf', 'pooh', 'pumba')
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)
test_loss, test_acc = model.evaluate(test_generator)
print("\nTest Accuracy:", test_acc)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()
images, labels = next(test_generator)
predictions = model.predict(images)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i in range(8):
    ax = axes[i]
    ax.imshow(images[i])
    pred_label = np.argmax(predictions[i])
    true_label = np.argmax(labels[i])
    ax.set_title(f"Pred: {classes[pred_label]}\nTrue: {classes[true_label]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
