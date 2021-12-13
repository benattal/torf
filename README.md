# NeRF: Neural Radiance Fields

## TL;DR quickstart

Download data from [here](https://drive.google.com/file/d/1H-iBHA1blhl94NF-ZjKBW0xrP_Nqu-G7/view?usp=sharing) and place its contents in the `./data` folder

To setup a conda environment and begin the training process:
```
conda env create -f environment.yml
conda activate tof_nerf
./scripts/real_fast.sh 30 seq_color21_small ./data 0 60
```

## Dynamic sequences

For the dino, and iOS sequences, run

```
./scripts/ios_fast.sh 30 dishwasher ./data
```

and 

```
./scripts/dino_fast.sh 30 ./data
```

respectively.

## Static sequences

For the bathroom sequences run

```
./scripts/mitsuba_static_bathroom.sh ./data 2 0,8
```

and

```
./scripts/mitsuba_static_bathroom_nerf.sh ./data 2 0,8
```