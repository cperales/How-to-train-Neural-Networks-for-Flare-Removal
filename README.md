# How-to-train-Neural-Networks-for-Flare-Removal

This code is a fork of Wu et al. 2021 code, placed in [Google-Research/Flare-Removal](https://github.com/google-research/google-research/tree/master/flare_removal). Same licence as in the original Google-Research repo.

Main modifications in this fork:
- It adds a `Dockerfile` and `requirements_docker.txt`, so you can run it although code uses `tensorflow_addons`, a library [no longer maintained](https://github.com/tensorflow/addons/issues/2807).
- Training resolution could be higher than 512. If you want to test full images, it is recommended to select the dimension of the maximum value between the height and the width of the images.


## Build and run the docker with GPU

You can build the docker by running

```
docker build --pull --rm -f "Dockerfile" -t flareremoval:tensorflow "."
```

and run it with GPUs with the following command

```
docker run -it --gpus all -v .:/repo flareremoval:tensorflow
```


# Training

Example of training:

```
python flare_removal/train.py --train_dir=train_logs --scene_dir=train_images/scenes --flare_dir=train_images/flares --val_scene_dir=val_images/scenes --ckpt_period=1 --val_fla
re_dir=val_images/flares --epochs=1 --training_res=512 --batch_size=2 --flare_loss_weight=0
```
