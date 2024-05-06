import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow_addons.layers import GraphAttention
from tensorflow_addons.losses import contrastive_loss

import numpy as np
import os
from PIL import Image

class ImageGraphSequence(Sequence):
    def __init__(self, image_dir, graph_dir, batch_size):
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.graph_files = sorted([os.path.join(graph_dir, f) for f in os.listdir(graph_dir) if f.endswith('.npy')])
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_images = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_graphs = self.graph_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = np.array([np.expand_dims(tf.image.resize(np.array(Image.open(img)), (224, 224)), axis=0)
                           for img in batch_images])
        graphs = [np.load(graph) for graph in batch_graphs]
        labels = np.zeros(len(images))  # Placeholder labels; adjust based on your needs
        return [images, graphs], labels

class SiameseGNN(Model):
    def __init__(self):
        super(SiameseGNN, self).__init__()
        self.convnet = tf.keras.Sequential([
            Conv2D(64, 5, activation='relu'),
            MaxPooling2D(),
            Conv2D(128, 5, activation='relu'),
            MaxPooling2D(),
            Flatten()
        ])
        self.gnn = GraphAttention(128, use_bias=True, kernel_initializer='glorot_uniform')
        self.fc = tf.keras.Sequential([
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(2)
        ])

    def call(self, inputs):
        img, graph = inputs
        cnn_features = self.convnet(img)
        gnn_features = self.gnn(graph)
        combined_features = tf.concat([cnn_features, gnn_features], axis=1)
        return self.fc(combined_features)

    def forward_once(self, img, graph):
        return self.call([img, graph])

    def forward(self, img1, graph1, img2, graph2):
        output1 = self.forward_once(img1, graph1)
        output2 = self.forward_once(img2, graph2)
        return output1, output2

def train(model, dataloader, optimizer, epochs=5):
    for epoch in range(epochs):
        for [img1, graph1], labels in dataloader:
            with tf.GradientTape() as tape:
                output1, output2 = model(img1, graph1)
                loss = contrastive_loss(output1, output2, labels)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# Adjusted paths for your dataset





image_dir = 'content/drive/MyDrive/Colabdata/Fluo-C2DL-Huh7-Training/Fluo-C2DL-Huh7/01'
graph_dir = 'content/drive/MyDrive/Colabdata/Fluo-C2DL-Huh7-Training/Fluo-C2DL-Huh7/01graph'

dataset = ImageGraphSequence(image_dir, graph_dir, batch_size=2)
model = SiameseGNN()
optimizer = Adam(learning_rate=0.0005)

train(model, dataset, optimizer)
