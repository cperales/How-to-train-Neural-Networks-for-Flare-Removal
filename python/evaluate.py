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
import synthesis
from time import perf_counter
import utils


flags.DEFINE_string(
    'eval_dir', '/tmp/eval',
    'Directory where evaluation summaries and outputs are written.')
flags.DEFINE_string(
    'train_dir', '/tmp/train',
    'Directory where training checkpoints are written. This script will '
    'repeatedly poll and evaluate the latest checkpoint.')
flags.DEFINE_string('scene_dir', None,
                    'Full path to the directory containing scene images.')
flags.DEFINE_string('flare_dir', None,
                    'Full path to the directory containing flare images.')
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
flags.DEFINE_list('training_res', [512, 512], 'Training resolution.')
FLAGS = flags.FLAGS


def main(_):
  eval_dir = FLAGS.eval_dir
  assert eval_dir, 'Flag --eval_dir must not be empty.'
  eval_dir_images = os.path.join(eval_dir, 'images')
  eval_dir_tfrecord = os.path.join(eval_dir, 'test')
  os.makedirs(eval_dir_images, exist_ok=True)
  os.makedirs(eval_dir_tfrecord, exist_ok=True)
  train_dir = FLAGS.train_dir
  assert train_dir, 'Flag --train_dir must not be empty.'
  summary_dir = os.path.join(eval_dir, 'summary')

  # Check training resolution
  if len(FLAGS.training_res) == 1:
    r = FLAGS.training_res[0]
    training_res = 2 * [int(r)]
  else:
    training_res = [int(r) for r in FLAGS.training_res]

  # Load data.
  scenes = data_provider.get_scene_dataset(
      FLAGS.scene_dir, FLAGS.data_source, FLAGS.batch_size, repeat=0,
      input_shape=(682, 1024, 3), shuffle=False)
  flares = data_provider.get_flare_dataset(FLAGS.flare_dir, FLAGS.data_source,
                                           FLAGS.batch_size)

  # Make a model.
  model = models.build_model(FLAGS.model, training_res)
  loss_fn = losses.get_loss(FLAGS.loss)

  ckpt = tf.train.Checkpoint(
      step=tf.Variable(0, dtype=tf.int64),
      training_finished=tf.Variable(True, dtype=tf.bool),
      model=model)

  summary_writer = tf.summary.create_file_writer(summary_dir)

  # The checkpoints_iterator keeps polling the latest training checkpoints,
  # until:
  #   1) `timeout` seconds have passed waiting for a new checkpoint; and
  #   2) `timeout_fn` (in this case, the flag indicating the last training
  #      checkpoint) evaluates to true.
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
    val_loss_list = list()
    val_psnr_list = list()
    val_ssim_list = list()
    for scene, flare in tf.data.Dataset.zip((scenes, flares)):
      loss_value, summary_dict = synthesis.run_step(
          scene,
          flare,
          model,
          loss_fn,
          noise=FLAGS.scene_noise,
          flare_max_gain=FLAGS.flare_max_gain,
          flare_loss_weight=FLAGS.flare_loss_weight,
          training_res=training_res)
      summary = tf.concat([summary_dict['combined'],
                           summary_dict['pred_scene'],
                           summary_dict['scene'],
                           summary_dict['pred_flare'],
                           summary_dict['flare']],
                    axis=2)
      val_loss_list.append(loss_value)
      val_psnr_list.append(summary_dict['psnr'])
      val_ssim_list.append(summary_dict['ssim'])
      with summary_writer.as_default():
        tf.summary.scalar('loss/val', loss_value, step=ckpt.step)
        tf.summary.image('prediction/val', summary, max_outputs=1, step=ckpt.step)
        for i in range(FLAGS.batch_size):
          counter += 1
          
          combined_i = summary_dict['combined'][i]
          image_i = tf.concat([combined_i,
                               summary_dict['pred_scene'][i],
                               summary_dict['scene'][i]],
                    axis=1)

          utils.save_image(image_i, eval_dir_images, str(counter) + '_combined.jpg')
          utils.save_tensor(combined_i, f"{eval_dir_tfrecord}/{str(counter)}.tfrecord")
      
  val_loss = tf.reduce_mean(tf.stack(val_loss_list))
  val_psnr = tf.reduce_mean(tf.stack(val_psnr_list))
  val_ssim = tf.reduce_mean(tf.stack(val_ssim_list))
  logging.info('%i images, avg values:', counter)
  logging.info('Val loss: %.4f, val psnr: %.4f, val ssim: %.4f', val_loss, val_psnr, val_ssim)
  logging.info('Done!')
  end = perf_counter()
  time_elapsed = end - start
  minutes = time_elapsed // 60
  sec = time_elapsed - minutes * 60
  logging.info('Elapsed %i minutes with %i seconds', minutes, sec)


if __name__ == '__main__':
  app.run(main)
