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
import models
import synthesis
from time import perf_counter
import utils


flags.DEFINE_string(
    'out_dir', '/tmp/out',
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
flags.DEFINE_integer('batch_size', 2, 'Evaluation batch size.')
flags.DEFINE_list('training_res', [512, 512], 'Training resolution.')
FLAGS = flags.FLAGS


def main(_):
  out_dir = FLAGS.out_dir
  assert out_dir, 'Flag --out_dir must not be empty.'
  image_dir = FLAGS.image_dir
  assert image_dir, 'Flag --image_dir must not be empty.'
  os.makedirs(out_dir, exist_ok=True)
  train_dir = FLAGS.train_dir
  assert train_dir, 'Flag --train_dir must not be empty.'

  # Check training resolution
  if len(FLAGS.training_res) == 1:
    r = FLAGS.training_res[0]
    training_res = 2 * [int(r)]
  else:
    training_res = [int(r) for r in FLAGS.training_res]

  # Load data.
  images = data_provider.get_scene_dataset(
      FLAGS.image_dir, FLAGS.data_source, FLAGS.batch_size, repeat=0, shuffle=False,
      input_shape=(682, 1024, 3))

  # Make a model.
  model = models.build_model(FLAGS.model, training_res)

  ckpt = tf.train.Checkpoint(
      step=tf.Variable(0, dtype=tf.int64),
      training_finished=tf.Variable(True, dtype=tf.bool),
      model=model)

  for ckpt_path in tf.train.checkpoints_iterator(
      train_dir, timeout=2, timeout_fn=lambda: ckpt.training_finished):
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
      pred_scene, combined, pred_flare = synthesis.run_step_wo_flare(
          image,
          model,
          training_res=training_res)
      for i in range(FLAGS.batch_size):
        counter += 1
        image_i = tf.concat([combined[i],
                             pred_scene[i],
                             pred_flare[i]],
                  axis=1)

        utils.save_image(image_i, out_dir, str(counter) + '_combined.jpg')
    break


  logging.info('%i images', counter)
  logging.info('Done!')
  end = perf_counter()
  time_elapsed = end - start
  minutes = time_elapsed // 60
  sec = time_elapsed - minutes * 60
  logging.info('Elapsed %i minutes with %i seconds', minutes, sec)


if __name__ == '__main__':
  app.run(main)
