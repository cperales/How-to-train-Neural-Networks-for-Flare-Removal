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


flags.DEFINE_string(
    'train_dir', '/tmp/train',
    'Directory where training checkpoints and summaries are written.')
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


@tf.function
def train_step(model, scene, flare, loss_fn, optimizer, training_res):
  """Executes one step of gradient descent."""
  with tf.GradientTape() as tape:
    loss_value, summary_dict = synthesis.run_step(
        scene,
        flare,
        model,
        loss_fn,
        noise=FLAGS.scene_noise,
        flare_max_gain=FLAGS.flare_max_gain,
        flare_loss_weight=FLAGS.flare_loss_weight,
        training_res=training_res)
  grads = tape.gradient(loss_value, model.trainable_weights)
  grads, _ = tf.clip_by_global_norm(grads, 5.0)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss_value, summary_dict


def val_step(model, n_scenes, val_scenes, val_flares, loss_fn, training_res):
  val_loss = list()
  val_psnr = list()
  val_ssim = list()

  counter = 0
  for val_scene, val_flare in tf.data.Dataset.zip((val_scenes, val_flares)):
    counter += 1
    loss_value, summary_dict = synthesis.run_step(
          val_scene,
          val_flare,
          model,
          loss_fn,
          noise=FLAGS.scene_noise,
          flare_max_gain=FLAGS.flare_max_gain,
          flare_loss_weight=FLAGS.flare_loss_weight,
          training_res=training_res)
    val_loss.append(loss_value)
    val_psnr.append(summary_dict['psnr'])
    val_ssim.append(summary_dict['ssim'])
    if counter == n_scenes:
      break

  return tf.reduce_mean(tf.stack(val_loss)), \
         tf.reduce_mean(tf.stack(val_psnr)), \
         tf.reduce_mean(tf.stack(val_ssim)), \
         summary_dict


def main(_):
  train_dir = FLAGS.train_dir
  assert train_dir, 'Flag --train_dir must not be empty.'
  summary_dir = os.path.join(train_dir, 'summary')
  model_dir = os.path.join(train_dir, 'model')

  # Check training resolution
  if len(FLAGS.training_res) == 1:
    r = FLAGS.training_res[0]
    training_res = 2 * [int(r)]
  else:
    training_res = [int(r) for r in FLAGS.training_res]

  # Load train data.
  scenes = data_provider.get_scene_dataset(
      FLAGS.scene_dir, FLAGS.data_source, FLAGS.batch_size, repeat=FLAGS.epochs,
      input_shape=(682, 1024, 3))
  flares = data_provider.get_flare_dataset(FLAGS.flare_dir, FLAGS.data_source,
                                           FLAGS.batch_size)
  
  if FLAGS.ckpt_period == 0:
    FLAGS.ckpt_period = len(os.listdir(FLAGS.scene_dir)) // FLAGS.batch_size
  
  # Load val data.
  if FLAGS.val_scene_dir:
    val_scenes = data_provider.get_scene_dataset(
        FLAGS.val_scene_dir, FLAGS.data_source, FLAGS.batch_size, input_shape=(682, 1024, 3))
    if FLAGS.val_flare_dir == FLAGS.flare_dir:
      val_flares = flares
    else:
      val_flares = data_provider.get_flare_dataset(FLAGS.val_flare_dir, FLAGS.data_source,
                                            FLAGS.batch_size)
    n_val_scenes = len(os.listdir(FLAGS.val_scene_dir)) // FLAGS.batch_size
    validation = True
  else:
    validation = False
    n_val_scenes = 0

  # Make a model.
  model = models.build_model(FLAGS.model, training_res)
  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
  loss_fn = losses.get_loss(FLAGS.loss)

  # Model checkpoints. Checkpoints don't contain model architecture, but
  # weights only. We use checkpoints to keep track of the training progress.
  ckpt = tf.train.Checkpoint(
      step=tf.Variable(0, dtype=tf.int64),
      training_finished=tf.Variable(False, dtype=tf.bool),
      optimizer=optimizer,
      model=model)
  ckpt_mgr = tf.train.CheckpointManager(
      ckpt, train_dir, max_to_keep=3, keep_checkpoint_every_n_hours=3)

  # Restore the latest checkpoint (model weights), if any. This is helpful if
  # the training job gets restarted from an unexpected termination.
  latest_ckpt = ckpt_mgr.latest_checkpoint
  restore_status = None
  if latest_ckpt is not None:
    # Note that due to lazy initialization, not all checkpointed variables can
    # be restored at this point. Hence 'expect_partial()'. Full restoration is
    # checked in the first training step below.
    restore_status = ckpt.restore(latest_ckpt).expect_partial()
    logging.info('Restoring latest checkpoint @ step %d from: %s', ckpt.step,
                 latest_ckpt)
  else:
    logging.info('Previous checkpoints not found. Starting afresh.')

  summary_writer = tf.summary.create_file_writer(summary_dir)

  step_time_metric = tf.keras.metrics.Mean('step_time')
  step_start_time = time.time()

  train_loss_list = list()
  train_psnr_list = list()
  train_ssim_list = list()
  for scene, flare in tf.data.Dataset.zip((scenes, flares)):
    # Perform one training step.
    loss_value, summary = train_step(model, scene, flare, loss_fn, optimizer, training_res)

    # Add the values
    train_loss_list.append(loss_value)
    train_psnr_list.append(summary['psnr'])
    train_ssim_list.append(summary['ssim'])

    # By this point, all lazily initialized variables should have been
    # restored by the checkpoint if one was available.
    if restore_status is not None:
      restore_status.assert_consumed()
      restore_status = None

    # Record elapsed time in this training step.
    step_end_time = time.time()
    step_time_metric.update_state(step_end_time - step_start_time)
    step_start_time = step_end_time

    # Write training summaries and checkpoints to disk.
    ckpt.step.assign_add(1)
    if ckpt.step % FLAGS.ckpt_period == 0:
      # Write model checkpoint to disk.
      ckpt_mgr.save()

      train_loss = tf.reduce_mean(tf.stack(train_loss_list))
      train_loss_list.clear()
      train_psnr = tf.reduce_mean(tf.stack(train_psnr_list))
      train_psnr_list.clear()
      train_ssim = tf.reduce_mean(tf.stack(train_ssim_list))
      train_ssim_list.clear()

      # Also save the full model using the latest weights. To restore previous
      # weights, you'd have to load the model and restore a previously saved
      # checkpoint.
      tf.keras.models.save_model(model, model_dir, save_format='tf')

      # Apply model to validation
      
      if validation:
        val_loss, val_psnr, val_ssim, val_summary = val_step(model, n_val_scenes, val_scenes, val_flares, loss_fn, training_res)

        val_image_summary = tf.concat([val_summary['combined'],
                                       val_summary['pred_scene'],
                                       val_summary['scene'],
                                       val_summary['pred_flare'],
                                       val_summary['flare']],
                                      axis=2)

        logging.info('Step %i, train loss: %.4f, val loss: %.4f', ckpt.step, train_loss.numpy(), val_loss)
        logging.info('         train psnr: %.4f, val psnr: %.4f', train_psnr.numpy(), val_psnr)
        logging.info('         train ssim: %.4f, val ssim: %.4f', train_ssim.numpy(), val_ssim)

      # Write summaries to disk, which can be visualized with TensorBoard.
      train_image_summary = tf.concat([summary['combined'],
                                       summary['pred_scene'],
                                       summary['scene'],
                                       summary['pred_flare'],
                                       summary['flare']],
                                      axis=2)

      with summary_writer.as_default():
        tf.summary.image('prediction/train', train_image_summary, max_outputs=1, step=ckpt.step)
        tf.summary.scalar('loss/train', train_loss, step=ckpt.step)
        tf.summary.scalar('psnr/train', train_psnr, step=ckpt.step)
        tf.summary.scalar('ssim/train', train_ssim, step=ckpt.step)

        if validation:
          tf.summary.image('prediction/val', val_image_summary, max_outputs=1, step=ckpt.step)
          tf.summary.scalar('loss/val', val_loss, step=ckpt.step)
          tf.summary.scalar('psnr/val', val_psnr, step=ckpt.step)
          tf.summary.scalar('ssim/val', val_ssim, step=ckpt.step)


        tf.summary.scalar(
            'step_time', step_time_metric.result(), step=ckpt.step)
        step_time_metric.reset_state()

  ckpt.training_finished.assign(True)
  ckpt_mgr.save()
  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
