# TöRF: Time-of-Flight Radiance Fields for Dynamic Scene View Synthesis

# Environment setup

Download data from [here](https://drive.google.com/drive/folders/18QsJeCYjqtfYCtduzeDMuulgW6EpF4wO?usp=sharing) and place its contents in the `./data` folder

To set up a conda environment, run:

```
conda env create -f environment.yml
conda activate torf
```

# Training

## Quick start

To train model on one of our sequences, run

```
./scripts/real.sh seq_color21_small 0 30
```

The last two arguments specify first frame to use and the last frame to use

## Dynamic sequences

For the real and iOS sequences, run

```
./scripts/real.sh seq_color21_small 0 30
```

and 

```
./scripts/ios.sh dishwasher 30
```

respectively.

## Static sequences

For the static sequences run

```
./scripts/static.sh bathroom_static 2 <width> <height> 0,8 
```

Where the last argument specifies which views to use for training. Image dimensions are 512x512 for the bathroom sequence, and 640x360 for bedroom.

# Using Your Own Data

## Format

If your data contains ground truth ToF images, then you should use the `RealDataset` loader, or some variant of it, and the data should be formatted as follows.

```
cams/
    tof_intrinsics[.npy|.mat]
    tof_extrinsics[.npy|.mat]
    color_intrinsics[.npy|.mat]
    color_extrinsics[.npy|.mat]
color/
    [*.npy|*.mat]
tof/
    [*.npy|*.mat]
```

**Note that each ToF image should contain (real, imaginary) components of the measured ToF phasor**

## Additional inputs

You can also optionally include:

```
cams/
    relative_R[.npy|.mat]
    relative_T[.npy|.mat]
    depth_range[.npy|.mat]
    phase_offset[.npy|.mat]
```

where `relative_R` and `relative_T` specify the relative pose of the color camera and ToF sensor. If you do not include these, they are initialized to identity. They can also be optimized during training with `--use_relative_poses` and `--optimize_relative_pose`

## Depth

The file `depth_range` specifies a value that is twice unambiguous depth range of the ToF sensor, and `phase_offset` an offset between the measured phase and the true phase. The phase offset can also be optimized during training with `--optimize_phase_offset`

If metric depth, rather than ToF is available, then you can replace the `tof` folder with a `depth` folder. In this case, you should convert depth into ToF before training (see `IOSDataset`).

## Extrinsics

We expect extrinsics (world to camera) in SfM format (y down, z forward). Note that the if the extrinsics are not scale correct (i.e. do not match the scale of depth / ToF), then they should be optimized during training with ``--optimize_poses``

## Additional flags

If your dataset contains collocated ToF / depth and color, you can add the flag `--collocated_pose`


# Evaluation

## Static evaluation

For evaluation on static sequences, run:

```
./scripts/static_eval.sh bathroom_static 2 <width> <height> 0,8
```

And then

```
python compute_metrics eval/[expname] 30
```

# Citation

```
@article{attal2021torf,
  title={TöRF: Time-of-Flight Radiance Fields for Dynamic Scene View Synthesis},
  author={Attal, Benjamin and Laidlaw, Eliot and Gokaslan, Aaron and Kim, Changil and Richardt, Christian and Tompkin, James and O'Toole, Matthew},
  journal={Advances in neural information processing systems},
  volume={34},
  year={2021}
}
```