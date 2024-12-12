# coding=utf-8

"""Data creation script."""

import os.path

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import data_provider
import utils
import synthesis
from time import perf_counter


flags.DEFINE_string('scene_dir', None,
                    'Full path to the directory containing scene images.')
flags.DEFINE_string('flare_dir', None,
                    'Full path to the directory containing flare images.')
flags.DEFINE_string('image_dir', None,
                    'Full path to the directory containing scene + flare images.')
flags.DEFINE_enum(
    'data_source', 'jpg', ['tfrecord', 'jpg'],
    'Source of training data. Use "jpg" for individual image files, such as '
    'JPG and PNG images. Use "tfrecord" for pre-baked sharded TFRecord files.')
flags.DEFINE_integer('batch_size', 1, 'Evaluation batch size.')
flags.DEFINE_float(
    'scene_noise', 0.01,
    'Gaussian noise sigma added in the scene in synthetic data. The actual '
    'Gaussian variance for each image will be drawn from a Chi-squared '
    'distribution with a scale of scene_noise.')
flags.DEFINE_float(
    'flare_max_gain', 10.0,
    'Max digital gain applied to the flare patterns during synthesis.')
flags.DEFINE_float('flare_loss_weight', 1.0,
                   'Weight added on the flare loss (scene loss is 1).')
flags.DEFINE_integer('training_res', 512,
                     'Image resolution at which the network is trained.')
FLAGS = flags.FLAGS


def main(_):
    image_dir = FLAGS.image_dir
    assert image_dir, 'Flag --image_dir must not be empty.'
    os.makedirs(os.path.join(image_dir, 'scenes'), exist_ok=True)
    os.makedirs(os.path.join(image_dir, 'flares'), exist_ok=True)
    os.makedirs(os.path.join(image_dir, 'images'), exist_ok=True)

    # Load data.
    scenes = data_provider.get_scene_dataset(
        FLAGS.scene_dir, FLAGS.data_source, FLAGS.batch_size, repeat=0)
    flares = data_provider.get_flare_dataset(FLAGS.flare_dir, FLAGS.data_source,
                                            FLAGS.batch_size)

    start = perf_counter()
    counter = 0
    for scene, flare in tf.data.Dataset.zip((scenes, flares)):
        # scene_i = tf.expand_dims(scene, 0, name=None)
        # flare_i = tf.expand_dims(flare, 0, name=None)
        counter += 1
        scene_i, flare_i, combined_i, gamma_i = synthesis.add_flare_wo_crop(
        scene,
        flare,
        noise=FLAGS.scene_noise,
        flare_max_gain=FLAGS.flare_max_gain,
        apply_affine=False)
    
        utils.save_image(tensor=scene_i,
                folder=os.path.join(image_dir, 'scenes'),
                filename=str(counter) + '_flare.jpg')
        utils.save_image(tensor=flare_i,
                folder=os.path.join(image_dir, 'flares'),
                filename=str(counter) + '_scene.jpg')
        utils.save_image(tensor=combined_i,
                folder=os.path.join(image_dir, 'images'),
                filename=str(counter) + '_image.jpg')

    logging.info('Done!')
    end = perf_counter()
    time_elapsed = end - start
    minutes = time_elapsed // 60
    sec = time_elapsed - minutes * 60
    logging.info('Elapsed %i minutes with %i seconds', minutes, sec)


if __name__ == '__main__':
  app.run(main)
