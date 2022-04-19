train:
	python tools/train.py ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_subset.py --work-dir output/2022_03_26_generalize_no_augmentation --resume-from output/2022_03_26_generalize_no_augmentation/epoch_69.pth

train-multi: 
	./tools/dist_train.sh ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi.py 4 --work-dir output/generalize_chair_2022_03_31_multi_full --resume-from output/generalize_chair_2022_03_31_multi_full/epoch_6.pth

test-dir: 
	python input/mono_det_demo_custom.py /mmdetection3d/input/input_test /mmdetection3d/objectron_processed_chair_all/annotations/objectron_test.json ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi.py ./output/generalize_chair_2022_04_01_multi_full/epoch_9.pth --out-dir /mmdetection3d/output/predict_generalize_chair_2022_03_31_multi_full_test

test-demo:
	python -u demo/mono_det_demo.py /mmdetection3d/objectron_processed_chair_all/images/chair_batch-36_17_50.jpg  /mmdetection3d/objectron_processed_chair_all/annotations/objectron_train.json  ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi_print.py ./output/generalize_chair_2022_04_01_multi_full/epoch_9.pth --out-dir /mmdetection3d/output/print_feat_2022_04_12

benchmark:
	python tools/analysis_tools/benchmark.py ./input/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_objectron-mono3d_pwo_objectron_generalize_full_ds_multi.py ./output/generalize_chair_2022_04_01_multi_full/epoch_9.pth