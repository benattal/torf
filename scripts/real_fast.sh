 XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda python torf.py --config config/config_real.txt \
    --dataset_type real \
    --scan $1 \
    --expname real_$1_$3_dynamic_fast \
    --num_views $3 \
    --num_frames $3 \
    --tof_image_width 320 --tof_image_height 240 \
    --tof_scale_factor 1 \
    --color_image_width 640 --color_image_height 480 \
    --color_scale_factor 0.5 \
    --i_img 1000 \
    --i_save 20000 \
    --i_video 20000 \
    --view_step 1 --view_start $2 --total_num_views $3 \
    --render_extrinsics_file data/render_poses/spiral_21.npy \
    --reverse_render_extrinsics \
    --render_extrinsics_scale 1.1

