train:
	python tools/train.py ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_subset.py --work-dir output/2022_03_26_generalize_no_augmentation --resume-from output/2022_03_26_generalize_no_augmentation/epoch_69.pth

train-multi: 
	./tools/dist_train.sh ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi.py 4 --work-dir output/generalize_chair_2022_03_31_multi_full --resume-from output/generalize_chair_2022_03_31_multi_full/epoch_6.pth

test-dir: 
	python input/mono_det_demo_custom.py /mmdetection3d/input/input_test /mmdetection3d/objectron_processed_chair_all_filtered/annotations/objectron_test.json ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi.py ./output/generalize_chair_2022_04_01_multi_full/epoch_9.pth --out-dir /mmdetection3d/output/predict_generalize_chair_2022_04_26_multi_full_test

test-demo:
	python -u demo/mono_det_demo.py /mmdetection3d/objectron_processed_chair_all/images/chair_batch-36_17_00050.jpg  /mmdetection3d/objectron_processed_chair_all/annotations/objectron_train.json  ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi_print.py ./output/generalize_chair_2022_04_01_multi_full/epoch_9.pth --out-dir /mmdetection3d/output/print_feat_2022_04_12

test-trace:
	python -u model_export.py /mmdetection3d/objectron_processed_chair_all/images/chair_batch-36_17_00050.jpg  /mmdetection3d/objectron_processed_chair_all/annotations/objectron_train.json  ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi.py ./output/generalize_chair_2022_04_01_multi_full/epoch_9.pth --out-dir /mmdetection3d/output/test_tracing_2022_05_2

benchmark:
	python tools/analysis_tools/benchmark.py ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi.py ./output/generalize_chair_2022_04_01_multi_full/epoch_9.pth

print_feat:
	python -u demo/mono_det_demo.py /mmdetection3d/objectron_processed_chair_all/images/chair_batch-36_17_50.jpg  /mmdetection3d/objectron_processed_chair_all/annotations/objectron_train.json  ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi_print.py ./output/generalize_chair_2022_04_01_multi_full/epoch_9.pth --out-dir /mmdetection3d/output/print_feat_2022_04_20

overfit:
	python tools/train.py ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_subset_single_overfit_lr.py --work-dir output/overfit_single_class_2022_04_20

train-book_chair_multi:
	./tools/dist_train.sh ./input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes.py 8 --work-dir output/generalize_chair_book_2022_05_04_multi_full --resume-from output/generalize_chair_book_2022_05_03_multi_full/epoch_2.pth

inference-dir-chair-multi-test:
	python input/mono_det_demo_custom.py /mmdetection3d/input/input_test_chair /mmdetection3d/objectron_processed_chair_all/annotations/objectron_test.json ./input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes_inference.py output/generalize_chair_book_2022_05_04_multi_full/epoch_10.pth --out-dir /mmdetection3d/output/2202-05-05_inference_chair_multi_e10

inference-dir-book-multi-test:
	python input/mono_det_demo_custom.py /mmdetection3d/input/input_test_book /mmdetection3d/objectron_processed_book_all/annotations/objectron_test.json ./input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes_inference.py output/generalize_chair_book_2022_05_04_multi_full/epoch_10.pth --out-dir /mmdetection3d/output/2202-05-05_inference_book_multi_e10

inference-dir-book-multi-custom:
	python input/mono_det_demo_custom.py /mmdetection3d/input/inference_custom_book_chair/ /mmdetection3d/input/inference_custom_book_chair_v3/annotation.json ./input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes_inference.py output/generalize_chair_book_2022_05_04_multi_full/epoch_10.pth --out-dir /mmdetection3d/output/2202-05-05_inference_book_chair_multi_custom_e10

test-book_chair_multi:
	python tools/test.py ./input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes.py output/generalize_chair_book_2022_05_01_multi_full/epoch_1.pth --eval mAP
	
multi-test-book_chair_multi:
	./tools/dist_test.sh ./input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes.py output/generalize_chair_book_2022_05_03_multi_full/epoch_2.pth 8 --eval mAP
	
book_chair_single:
	python tools/train.py ./input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes.py --work-dir output/test_chair_book_2022_04_29_multi_full

book_chair_test:
	python tools/train.py ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_overfit_single_two_classes.py --work-dir output/test_chair_book_2022_04_29_multi_full
	
tteest:
	./tools/dist_train.sh ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_overfit_single_two_classes.py 8 --work-dir output/test_chair_book_2022_04_29_multi_full

last_test:
	python tools/test.py ./input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes.py output/generalize_chair_book_2022_05_01_multi_full/epoch_1.pth --eval mAP

deploy-model:
	python ./mmdeploy/tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --test-img ${TEST_IMG} \
    --work-dir ${WORK_DIR} \
    --calib-dataset-cfg ${CALIB_DATA_CFG} \
    --device ${DEVICE} \
    --log-level INFO \
    --show \
    --dump-info

video_inference:
	python input/mono_det_demo_custom.py /mmdetection3d/input/video_inference/... /mmdetection3d/input/video_inference/annotations.json ./input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes_inference.py output/generalize_chair_book_2022_05_04_multi_full/epoch_10.pth --out-dir /mmdetection3d/output/video_inference__...


bike_overfit:
	python tools/train.py ./input/smoke9D_objectron_generalize_full_ds_multigpu_bike.py --work-dir output/overfit_bike_2022_07_17_bike_v2

inference-dir-bike-overfit:
	python input/mono_det_demo_custom.py /mmdetection3d/input/bike_overfit_test /mmdetection3d/input/objectron_processed_bike_overfit/annotations/objectron_train.json ./input/smoke9D_objectron_generalize_full_ds_multigpu_bike.py output/test_chair_book_2022_07_16_bike/epoch_100.pth --out-dir /mmdetection3d/output/2202-07-17_inference_bike_overfit

inference-dir-bike-overfit-single:
	python -u demo/mono_det_demo.py /mmdetection3d/input/bike_overfit_test/bike_batch-1_6_00250.jpg /mmdetection3d/input/objectron_processed_bike_overfit/annotations/objectron_train.json ./input/smoke9D_objectron_generalize_full_ds_multigpu_bike.py output/overfit_bike_2022_07_17_bike_v2/epoch_75.pth --out-dir /mmdetection3d/output/2202-07-17_inference_bike_overfit_v2

train-bike_multi:
	./tools/dist_train.sh ./input/smoke9D_objectron_generalize_full_ds_multigpu_bike.py 4 --work-dir output/2022_07_18_generalize_bike_multi_full --resume-from output/2022_07_18_generalize_bike_multi_full/epoch_13.pth

bike_test:
	python tools/test.py ./input/smoke9D_objectron_generalize_full_ds_multigpu_bike.py output/2022_07_18_generalize_bike_multi_full/epoch_18.pth --eval mAP

bike_test_images:
	python input/mono_det_demo_custom.py /mmdetection3d/input/test_bike_images /mmdetection3d/objectron_processed_bike_all/annotations/objectron_test.json ./input/smoke9D_objectron_generalize_full_ds_multigpu_bike.py /mmdetection3d/output/2022_07_18_generalize_bike_multi_full/epoch_28.pth --out-dir /mmdetection3d/output/2202-07-18_inference_bike_test_images_e28

video-inference:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_11_19.MOV 790 790 960 720 363 475 2
video-inference_12_15:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_12_15.MOV 790 790 960 720 360 445 2

video-inference_11_10:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_11_10.MOV 780 780 960 720 360 465 2

video-inference_8_16:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_8_16.MOV 790 790 960 720 360 490 2

video-inference_6_11:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_6_11.MOV 776 776 960 720 358 469 2

video-inference_1_47:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_1_47.MOV 730 730 960 720 360 469 2

video-inference_1_25:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_1_25.MOV 730 730 960 720 360 465 2

video-inference_11_26:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_11_26.MOV 783 783 960 720 358 469 2

video-inference_10_09:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_10_9.MOV 781 781 960 720 358 469 2

video-inference_3_18:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_3_18.MOV 781 781 960 720 358 469 2

video-inference_2_46:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_2_46.MOV 789 789 960 720 358 469 2

video-inference_2_36:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_2_36.MOV 789 789 960 720 358 469 2

video-inference_1_34:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_1_34.MOV 781 781 960 720 358 469 2

video-inference_1_15:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_1_15.MOV 781 781 960 720 358 469 2

video-inference_1_13:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_1_13.MOV 781 781 960 720 358 469 2

video-inference_0_45:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_0_45.MOV 781 781 960 720 358 469 2

video-inference_12_6:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_12_6.MOV 781 781 960 720 358 469 2

video-inference_11_19:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_11_19.MOV 781 781 960 720 358 469 2

video-inference_1_6:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_1_6.MOV 781 781 960 720 358 469 2

video-inference_list:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_12_15.MOV 790 790 960 720 360 445 2
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_11_10.MOV 780 780 960 720 360 465 2
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_1_47.MOV 730 730 960 720 360 469 2
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/motor_2_36.MOV 789 789 960 720 358 469 2
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_1_6.MOV 781 781 960 720 358 469 2


video-inference_mb_1:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld/2022-07-20_mb_1.mov 1447 1447 1920 1440 720 925 1

video-inference_mb_2:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld/2022-07-20_mb_2.mov 1447 1447 1920 1440 720 925 1

video-inference_mb_3:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld/2022-07-20_mb_3.mov 1447 1447 1920 1440 720 925 1

video-inference_mb_4:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld/2022-07-20_mb_4.mov 1447 1447 1920 1440 720 925 1

video-inference_mb_6:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld/2022-07-20_mb_6.mov 1447 1447 1920 1440 720 925 1

video-inference_cb_1:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld/2022-07-20_cb_1.mov 1447 1447 1920 1440 720 925 1

video-inference_cb_2:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld/2022-07-20_cb_2.mov 1447 1447 1920 1440 720 925 1

video-inference_cb_3:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld/2022-07-20_cb_3.mov 1447 1447 1920 1440 720 925 1

video-inference_cb_4:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld/2022-07-20_cb_4.mov 1447 1447 1920 1440 720 925 1

video-inference_double_1:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_double_1.mov 1447 1447 1920 1440 720 925 1

video-inference_double_2:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_double_2.mov 1447 1447 1920 1440 720 925 1

video-inference_standing_2:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_standing_2.mov 1447 1447 1920 1440 720 925 1

video-inference_standing_3:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_standing_3.mov 1447 1447 1920 1440 720 925 1

video-inference_standing_4:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_standing_4.mov 1447 1447 1920 1440 720 925 1

video-inference_standing_5:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_standing_5.mov 1447 1447 1920 1440 720 925 1

video-inference_standing_6:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_standing_6.mov 1447 1447 1920 1440 720 925 1

video-inference_driving_3:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_driving_3.mov 1447 1447 1920 1440 720 925 1

video-inference_driving_4:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_driving_4.mov 1447 1447 1920 1440 720 925 1

video-inference_driving_5:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_driving_5.mov 1447 1447 1920 1440 720 925 1

video-inference_driving_1:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_driving_1.mov 1447 1447 1920 1440 720 925 1

video-inference_driving_2:
	python -m viz.app.plot_trajectory video_animation_9d ./input/inthewilld2/2022-07-20_driving_2.mov 1447 1447 1920 1440 720 925 1

parameters:
	(mov_filename, f_x=1492, f_y=1492, h=1920, w=1440, o_x=720, o_y=960)


	bike_11_19: [790.2564697265625, 0.0, 363.83502197265625], [0.0, 790.2564697265625, 475.51116943359375], [0.0, 0.0, 1.0]

smoke_original:
	python input/mono_det_demo_custom.py /mmdetection3d/input/car_test_images ./input/car_test_images/annotation.json ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-07-20_inference_car_images

smoke_single:
	python -u demo/mono_det_demo.py /mmdetection3d/input/car_test_images/ffmpeg_23.png ./input/car_test_images/annotation.json ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-07-20_inference_car_single

smoke_single_r:
	python -u demo/mono_det_demo.py /mmdetection3d/input/car_test_images_rescaled/rescaled_1.png ./input/car_test_images_rescaled/annotation_rescaled.json ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-07-20_inference_car_single_r

smoke_intrinsics:
	python input/mono_det_demo_custom.py /mmdetection3d/input/car_test_images ./input/car_test_images/annotation_adapted.json ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-07-20_inference_car_images_intrinsics

smoke_intrinsics_slk:
	python input/mono_det_demo_custom.py /mmdetection3d/input/slk ./input/slk/annotation.json ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-08-08_inference_car_images_intrinsics

smoke_intrinsics_slk_rs:
	python input/mono_det_demo_custom.py /mmdetection3d/input/slk_rs ./input/slk_rs/annotation.json ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-08-08_inference_car_images_intrinsics_rs

smoke_kitti:
	python input/mono_det_demo_custom.py /mmdetection3d/input/kitti ./input/kitti/annotation.json ./config/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/config/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-08-08_inference_kitti

smoke_nuscenes:
	python input/mono_det_demo_custom.py /mmdetection3d/input/nuscenes ./input/nuscenes/annotation.json ./configsmoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/config/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-08-08_inference_nuscenes

smoke_ls:
	python input/mono_det_demo_custom.py /mmdetection3d/input/landscape_test_car ./input/landscape_test_car/annotation.json ./config/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/config/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-08-10_inference_custom

smoke_cmp:
	python input/mono_det_demo_custom.py /mmdetection3d/input/cmp_images ./input/cmp_images/annotation.json ./config/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/config/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-08-10_inference_cmp

smoke_pd:
	python input/mono_det_demo_custom.py /mmdetection3d/input/padded_cmp ./input/padded_cmp/annotation.json ./config/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py /mmdetection3d/config/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth --out-dir /mmdetection3d/output/2202-08-12_inference_cmp_pd

video_chair:
	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_1_6.MOV 781 781 960 720 358 469 2


human_keypoint_estimation:
	python -m perception_pipeline.instructar detect_humans ./input_data/car_multi_person_1.mov keyp 1 1 25

hpe:
	python -m perception_pipeline.detections_human_pose detect ./input_data/car_multi_person_1.mov keyp 1 1 5

new_smoke:
	python -m perception_pipeline.instructar detect_cars ./input_data/test_video_car.mov --kitti=0 1 1 100 config checkpoint
	python -m perception_pipeline.instructar detect_cars ./input_data/car_multi_person_1.mov --kitti=1 1 1 100 config checkpoint

	python -m viz.app.plot_trajectory video_animation_9d ./input/bike_videos/bike_1_6.MOV 781 781 960 720 358 469 2



new_pipeline:

	python -m perception_pipeline.instructar detect_humans ./input_data/car_multi_person_1.mov keyp 1 1 100
	python -m perception_pipeline.instructar detect_cars ./input_data/car_multi_person_1.mov --kitti=1 1 1 100 config checkpoint

	python -m perception_pipeline.instructar merge_detections --car_dir=./output2/20220818_085015-car_multi_person_1-cars  --human_dir=./output2/20220818_085337-car_multi_person_1-humans
	python -m perception_pipeline.instructar merge_detections --car_dir=./output2/20220818_092825-car_multi_person_5-cars  --human_dir=./output2/20220818_094526-car_multi_person_5-humans
	python -m perception_pipeline.instructar merge_detections --car_dir=./output2/20220818_092834-car_multi_person_6-cars  --human_dir=./output2/20220818_094534-car_multi_person_6-humans
	python -m perception_pipeline.instructar merge_detections --car_dir=./output2/20220818_092844-car_multi_person_7-cars  --human_dir=./output2/20220818_094553-car_multi_person_7-humans

	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220818_021737-zurich_traffic_1-cars  --human_dir=./output3/20220818_021716-zurich_traffic_1-humans

	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220818_022238-zurich_traffic_2-cars  --human_dir=./output3/20220818_022253-zurich_traffic_2-humans

	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220818_024006-zurich_traffic_3-cars  --human_dir=./output3/20220818_024026-zurich_traffic_3-humans

	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220818_032552-car_multi_person_1-cars  --human_dir=./output3/20220818_032927-car_multi_person_1-humans

	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220819_090750-car_multi_person_7-cars  --human_dir=./output3/20220819_090423-car_multi_person_7-humans

	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220819_024441-car_multi_person_1-cars  --human_dir=./output3/20220819_024439-car_multi_person_1-humans

	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220822_094944-car_multi_person_1-cars  --human_dir=./output3/20220822_095011-car_multi_person_1-humans

	20220818_022519-zurich_traffic_4-humans
	20220818_022523-zurich_traffic_4-cars

	20220818_024006-zurich_traffic_3-cars
	20220818_024026-zurich_traffic_3-humans
	 


	python -m perception_pipeline.instructar visualize_3D --input_file=./output2/20220817_084426-car_multi_person_1-scene_frames.json --frame_index=0


	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220823_041926-car_multi_person_6-scene_frames.json


	python -m perception_pipeline.instructar detect_cars ./input_data/lmulti.mov --kitti=1 1 1 100 config checkpoint


velocity_plots:
	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220822_121230-car_multi_person_1-scene_frames.json
	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220823_040817-car_multi_person_2-scene_frames.json
	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220822_055230-car_multi_person_3-scene_frames.json
	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220823_023616-car_multi_person_4-scene_frames.json
	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220823_041458-car_multi_person_5-scene_frames.json
	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220823_041926-car_multi_person_6-scene_frames.json
	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220822_091445-car_multi_person_7-scene_frames.json

test:
	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220822_091445-car_multi_person_7-scene_frames.json
	python -m perception_pipeline.instructar extract_keyframes --input_file=./output3/20220907_103322-20_standing_6-scene_frames.json


track_bbox_bike:
	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input/bike_videos/bike_12_15.MOV --fps_divisor=60
	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input/inthewilld2/2022-07-20_standing_6.mov --fps_divisor=60

	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220905_085538-2022-07-20_standing_6-objects  --human_dir=./output3/20220905_090215-2022-07-20_standing_6-humans

	python -m perception_pipeline.instructar detect_humans ./input/inthewilld2/2022-07-20_standing_6.mov keyp 1 0 60

track_bbox_chair:
	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input_data/test_chair_tripod_1.mov --fps_divisor=30 --fobject=chair

	python -m perception_pipeline.instructar detect_humans ./input/inthewilld2/2022-07-20_standing_6.mov keyp 0 0 10 1
	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input/inthewilld2/2022-07-20_standing_6.mov --fps_divisor=60



	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input/inthewilld/2022-07-20_mb_6.mov --fps_divisor=60

	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input_data/test_static_chair_3.mov --fps_divisor=60


	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220907_101353-2022-07-20_standing_6-objects --human_dir=./output3/20220907_101428-2022-07-20_standing_6-humans

	python -m perception_pipeline.instructar merge_detections --car_dir=./output3/20220907_125309-human_chair_1-objects --human_dir=./output3/20220907_125955-human_chair_1-humans

	
	

	

	python -m perception_pipeline.instructar detect_humans ./input_data/human_chair_1.mov --mode= keyp --custom=0 --landscape=0 --fps_divisor=60 --visualize=1

	python -m perception_pipeline.instructar merge_detections --object_dir=./output4/20220913_095308-human_chair_1-objects --human_dir=./output4/20220913_101044-human_chair_1-humans

	python -m perception_pipeline.instructar extract_keyframes_objectron --input_file=./output4/20220913_102719-human_chair_1-scene_frames.json --fps=60 --bbox_frame_index=500 --human_indices=[1,2]


	visualize broken chair to human
	python -m  viz.app.plot_trajectory visualize_3D --input_file=./viz/data/20220907_011137-human_chair_1-scene_frames.json --frame_index=0 --focus_index=0 --kitti=0

	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input/inthewilld2/2022-07-20_double_1.mov --fps_divisor=60


	python -m perception_pipeline.instructar extract_keyframes_objectron --input_file=./output3/20220907_103322-20_standing_6-scene_frames.json --fps=60 --bbox_frame_index=0 --human_indices=[1,2]

	python -m perception_pipeline.instructar extract_keyframes_objectron --input_file=./output4/20220913_102719-human_chair_1-scene_frames.json --fps=60 --bbox_frame_index=0 --human_indices=[1,2]




	python -m perception_pipeline.instructar merge_detections --object_dir=./output4/20220920_014225-2022-07-20_standing_6-objects --human_dir=./output3/20220907_101428-2022-07-20_standing_6-humans
	python -m perception_pipeline.instructar detect_humans ./input/inthewilld2/2022-07-20_standing_6.mov --mode= keyp --custom=0 --landscape=0 --fps_divisor=60 --visualize=1

	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input/inthewilld2/2022-07-20_standing_6.mov --fps_divisor=1 --fobject=bike
	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input_data/test_setup.mov --fps_divisor=30 --fobject=chair

	python -m perception_pipeline.instructar extract_keyframes_objectron --input_file=./output4/20220913_102719-human_chair_1-scene_frames.json --fps=60 --bbox_frame_index=0 --human_indices=[1,] --frame_idx=660

	python -m perception_pipeline.instructar extract_keyframes_objectron --input_file=./output4/20220920_015241-20_standing_6-scene_frames.json --fps=60 --bbox_frame_index=0 --human_indices=[1,] --frame_idx=0


	python -m perception_pipeline.instructar detect_humans ./input_data/human_chair_interaction_1.mov --mode= keyp --custom=0 --landscape=0 --fps_divisor=1 --visualize=0

	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input_data/human_chair_interaction_1.mov --fps_divisor=1 --fobject=chair


	python -m perception_pipeline.instructar detect_humans ./input_data/human_chair_interaction_1.mov --mode=keyp --custom=0 --landscape=0 --fps_divisor=100 --visualize=1

	
	python -m perception_pipeline.instructar merge_detections --object_dir=./output4/20220921_025429-human_chair_interaction_1-objects --human_dir=./output4/20220921_025232-human_chair_interaction_1-humans

	

	python -m perception_pipeline.instructar detect_objectron_objects --mov_filename=./input_data/publi_bike_ls.mov --fps_divisor=30 --fobject=bike
	
	python -m perception_pipeline.instructar extract_keyframes_objectron --input_file=./output4/20220920_015241-20_standing_6-scene_frames.json --fps=60 --bbox_frame_index=0 --human_indices=[1,] --frame_idx=0