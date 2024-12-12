# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Training script for flare removal.

This script trains a model that outputs a flare-free image from a flare-polluted
image.
"""
import os.path
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np

import data_provider
import losses
import models
import synthesis
from time import perf_counter
import utils


flags.DEFINE_string(
    'eval_dir', '/tmp/eval',
    'Directory where evaluation summaries and outputs are written.')
flags.DEFINE_string('scene_dir', None,
                    'Full path to the directory containing scene images.')
flags.DEFINE_string('val_scene_dir', None,
                    'Full path to the directory containing validation scene images.')
flags.DEFINE_string('flare_dir', None,
                    'Full path to the directory containing flare images.')
flags.DEFINE_string('val_flare_dir', None,
                    'Full path to the directory containing validation flare images.')
flags.DEFINE_enum(
    'data_source', 'jpg', ['tfrecord', 'jpg'],
    'Source of training data. Use "jpg" for individual image files, such as '
    'JPG and PNG images. Use "tfrecord" for pre-baked sharded TFRecord files.')
flags.DEFINE_string('model', 'unet', 'the name of the training model')
flags.DEFINE_string('loss', 'percep', 'the name of the loss for training')
flags.DEFINE_integer('batch_size', 2, 'Training batch size.')
flags.DEFINE_integer('epochs', 100, 'Training config: epochs.')
flags.DEFINE_integer(
    'ckpt_period', 0,
    'Write model checkpoint and summary to disk every ckpt_period steps.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
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
# flags.DEFINE_integer('training_res', 512, 'Training resolution.')
flags.DEFINE_list('training_res', [512, 512], 'Training resolution.')
FLAGS = flags.FLAGS


def main(_):
  eval_dir = FLAGS.eval_dir
  assert eval_dir, 'Flag --eval_dir must not be empty.'
  eval_dir_images = os.path.join(eval_dir, 'images')
  os.makedirs(eval_dir_images, exist_ok=True)

  # Check training resolution
  if len(FLAGS.training_res) == 1:
    r = FLAGS.training_res[0]
    training_res = 2 * [int(r)]
  else:
    training_res = [int(r) for r in FLAGS.training_res]

  # Load train data.
  scenes = data_provider.get_scene_dataset(
      FLAGS.scene_dir, FLAGS.data_source, FLAGS.batch_size, repeat=FLAGS.epochs,
      input_shape=(682, 1024, 3), shuffle=False)
  flares = data_provider.get_flare_dataset(FLAGS.flare_dir, FLAGS.data_source,
                                           FLAGS.batch_size)

  counter = 0
  for scene, flare in tf.data.Dataset.zip((scenes, flares)):
    # Perform one training step.
    new_scene, new_flare, combined, gamma = \
        synthesis.add_flare(scene=scene, 
                            flare=flare,
                            noise=FLAGS.scene_noise,
                            flare_max_gain=FLAGS.flare_max_gain,
                            training_res=training_res)

    for i in range(FLAGS.batch_size):
      counter += 1
      image_i = combined[i]
      utils.save_image(image_i, eval_dir_images, str(counter) + '_combined.jpg')

  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
