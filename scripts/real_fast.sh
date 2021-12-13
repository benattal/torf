 XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda python tof_nerf.py --config config/config_fast.txt \
    --dataset_type real \
    --scan $2 \
    --datadir $3 \
    --expname real_$2_$1_dynamic_fast \
    --depth_range 15.0 --scene_scale 1.0 --falloff_range 8.0 \
    --min_depth_fac 0.05 --max_depth_fac 0.5 \
    --num_views $1 \
    --num_frames $1 \
    --tof_image_width 320 --tof_image_height 240 \
    --tof_scale_factor 1 \
    --color_image_width 640 --color_image_height 480 \
    --color_scale_factor 0.5 \
    --lrate 1e-3 --lrate_decay 250  \
    --lrate_calib 1e-3 --lrate_decay_calib 100 --lrate_calib_fac 0.1 \
    --i_img 1000 \
    --i_save 20000 \
    --i_video 20001 \
    --view_step 1 --view_start $4 --total_num_views $5 \
    --optimize_poses --pose_reg_weight 0.0 --use_relative_poses --optimize_relative_pose \
    --identity_pose_initialization \
    --static_scene_iters 5000 --calibration_pretraining \
    --model_reset_iters 5000 --reset_static_model \
    --dynamic --empty_weight 0.0 --empty_weight_decay 0.1 --empty_weight_decay_steps 500 \
    --latent_code_size 256 \
    --tof_weight 8.0 --tof_weight_decay 0.5 --tof_weight_decay_steps 125 \
    --radiance_weight 2.0 --no_radiance_iters 0 \
    --extrinsics_file data/render_poses/spiral_21.npy \
    --reverse_extrinsics \
    --extrinsics_scale 1.1 \
    #--double_transmittance \
    #--depth_range 10.0 --scene_scale 1.0 --falloff_range 5.0 \

