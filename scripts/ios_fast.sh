   XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda  python torf.py --config config/config_ios.txt \
    --dataset_type ios \
    --scan $1 \
    --expname ios_$1_$2_dynamic_fast \
    --num_views $2 \
    --num_frames $2 \
    --tof_image_width 384 --tof_image_height 288 \
    --tof_scale_factor 1 \
    --color_image_width 384 --color_image_height 288 \
    --color_scale_factor 1 \
    --i_img 1000 \
    --i_save 20000 \
    --i_video 20000 \
    --view_step 1 --view_start 0 --total_num_views 40 \
    --render_extrinsics_file data/render_poses/spiral_ios.npy \
    --reverse_render_extrinsics \
    --render_extrinsics_scale 1.1

