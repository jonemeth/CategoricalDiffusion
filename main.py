import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from categorical_diffusion import CategoricalDiffusion
from network import NormType
from utils import random_times


def create_dataset():
    print("Loading dataset", flush=True)
    print("tf.keras.backend.image_data_format(): ", tf.keras.backend.image_data_format(), flush=True)
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('int32')
    train_images = np.pad(train_images, [[0, 0], [2, 2], [2, 2], [0, 0]])
    train_images[train_images < 128] = 0
    train_images[train_images >= 128] = 1

    print("Building dataset", flush=True)
    n_train = train_images.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(train_images)\
        .shuffle(n_train, reshuffle_each_iteration=True)\

    print('datasets ready', flush=True)

    return dataset


def generate_and_save_images(model, out_filename):
    predictions = model.generate_images(64)

    predictions *= 255

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(out_filename)
    plt.close(fig)


def train(args):
    print(args)

    if os.path.exists(args.output_dir):
        if 0 < len(os.listdir(args.output_dir)):
            raise RuntimeError("Output directory is not empty!")
    else:
        os.makedirs(args.output_dir)

    dataset = create_dataset().batch(args.batch_size)
    model = CategoricalDiffusion(2, args)

    optimizer = tf.keras.optimizers.Adam(1e-4)

    loss_running = 0.0
    for epoch in range(args.epochs):
        for i,  batch in enumerate(dataset):

            t = random_times(batch.shape[0], model.time_steps)

            loss = model.compute_apply_gradients(optimizer, batch, tf.convert_to_tensor(t, dtype=tf.float32))

            loss = loss.numpy()
            loss_running = loss_running * 0.99 + loss * 0.01

            if 0 == i % 10:
                print(f"epoch: {epoch:4d} iter: {i:4d}/{len(dataset):4d} loss: {loss_running:.4f}", flush=True)

        out_filename = args.output_dir + f'/image_at_epoch_{epoch:04d}.png'
        generate_and_save_images(model, out_filename)


def main():
    parser = ArgumentParser()
    parser.add_argument('--schedule',
                        type=CategoricalDiffusion.Schedule,
                        choices=list(CategoricalDiffusion.Schedule),
                        default=CategoricalDiffusion.Schedule.EXPONENTIAL)
    parser.add_argument('--time_steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--norm_type', type=NormType, choices=list(NormType), default=NormType.GROUPNORM)
    parser.add_argument('--output_dir', type=str, default="samples")

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
