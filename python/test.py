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

"""Evaluation script for flare removal."""

import os.path

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import data_provider
import losses
import models
import utils
import synthesis
from time import perf_counter


flags.DEFINE_string(
    'test_dir', '/tmp/eval',
    'Directory where outputs are written.')
flags.DEFINE_string(
    'train_dir', '/tmp/train',
    'Directory where training checkpoints are written. This script will '
    'repeatedly poll and evaluate the latest checkpoint.')
flags.DEFINE_string('image_dir', None,
                    'Full path to the directory containing images.')
flags.DEFINE_enum(
    'data_source', 'jpg', ['tfrecord', 'jpg'],
    'Source of training data. Use "jpg" for individual image files, such as '
    'JPG and PNG images. Use "tfrecord" for pre-baked sharded TFRecord files.')
flags.DEFINE_string('model', 'unet', 'the name of the training model')
flags.DEFINE_string('loss', 'percep', 'the name of the loss for training')
flags.DEFINE_integer('batch_size', 2, 'Evaluation batch size.')
flags.DEFINE_float(
    'learning_rate', 1e-4,
    'Unused placeholder. The flag has to be defined to satisfy parameter sweep '
    'requirements.')
flags.DEFINE_float('flare_loss_weight', 1.0,
                   'Weight added on the flare loss (scene loss is 1).')
flags.DEFINE_integer('training_res', 512,
                     'Image resolution at which the network is trained.')
FLAGS = flags.FLAGS


def main(_):
  train_dir = FLAGS.train_dir
  assert train_dir, 'Flag --train_dir must not be empty.'
  test_dir = FLAGS.test_dir
  os.makedirs(os.path.join(test_dir, 'target'), exist_ok=True)
  os.makedirs(os.path.join(test_dir, 'pred'), exist_ok=True)

  # Load data.
  images = data_provider.get_scene_dataset(
      FLAGS.image_dir, FLAGS.data_source, FLAGS.batch_size, repeat=0)

  # Make a model.
  model = models.build_model(FLAGS.model, FLAGS.batch_size)
  loss_fn = losses.get_loss(FLAGS.loss)

  ckpt = tf.train.Checkpoint(
      step=tf.Variable(0, dtype=tf.int64),
      training_finished=tf.Variable(False, dtype=tf.bool),
      model=model)


  # The checkpoints_iterator keeps polling the latest training checkpoints,
  # until:
  #   1) `timeout` seconds have passed waiting for a new checkpoint; and
  #   2) `timeout_fn` (in this case, the flag indicating the last training
  #      checkpoint) evaluates to true.
  for ckpt_path in tf.train.checkpoints_iterator(
      train_dir, timeout=30, timeout_fn=lambda: ckpt.training_finished):
    try:
      status = ckpt.restore(ckpt_path)
      # Assert that all model variables are restored, but allow extra unmatched
      # variables in the checkpoint. (For example, optimizer states are not
      # needed for evaluation.)
      status.assert_existing_objects_matched()
      # Suppress warnings about unmatched variables.
      status.expect_partial()
      logging.info('Restored checkpoint %s @ step %d.', ckpt_path, ckpt.step)
    except (tf.errors.NotFoundError, AssertionError):
      logging.exception('Failed to restore checkpoint from %s.', ckpt_path)
      continue
    
    start = perf_counter()
    counter = 0
    for image in images:
      pred_image = synthesis.run_step_wo_flare(
          image,
          model,
          training_res=FLAGS.training_res)
      for i in range(FLAGS.batch_size):
        counter += 1
        utils.save_image(image[i],
                         folder=os.path.join(FLAGS.test_dir, "target"),
                         filename=str(counter) + '.jpg')
        utils.save_image(pred_image[i],
                         folder=os.path.join(FLAGS.test_dir, "pred"),
                         filename=str(counter) + '_pred.jpg')

  logging.info('Done!')
  end = perf_counter()
  time_elapsed = end - start
  minutes = time_elapsed // 60
  sec = time_elapsed - minutes * 60
  logging.info('Elapsed %i minutes with %i seconds', minutes, sec)


if __name__ == '__main__':
  app.run(main)
