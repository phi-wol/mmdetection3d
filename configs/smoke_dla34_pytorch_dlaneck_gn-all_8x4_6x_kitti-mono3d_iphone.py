_base_ = [
    '../configs/_base_/datasets/kitti-mono3d.py', '../configs/_base_/models/smoke.py',
    '../configs/_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='Adam', lr=2.5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup=None, step=[50])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=72)
log_config = dict(interval=10)

find_unused_parameters = True
class_names = ['Pedestrian', 'Cyclist', 'Car']
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RandomShiftScale', shift_scale=(0.2, 0.4), aug_prob=0.3),
    dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
            'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=(1440, 1920),
    #     flip=False,
    #     transforms=[
    #         dict(type='AffineResize', img_scale=(1440, 1920), down_ratio=5),
    #         dict(type='Pad', size=(1280, 384)),
    #     ]),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(530, 384), # (1440, 1920),
        flip=False,
        transforms=[
            #dict(type='Resize', img_scale=(288, 384)),
            #dict(type='AffineResize', img_scale=(288, 384), down_ratio=4),
            #dict(type='Pad', size=(1280, 384)),
            #dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
            dict(type='AffineResize', img_scale=(530, 384), down_ratio=4),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
        # dict(
        # type='MultiScaleFlipAug',
        # img_scale=(288, 384),
        # flip=False,
        # transforms=[
        #     #dict(type='Resize', img_scale=(288, 384)),
        #     #dict(type='AffineResize', img_scale=(288, 384), down_ratio=4),
        #     #dict(type='Pad', size=(1280, 384)),
        #     #dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
        #     dict(type='AffineResize', img_scale=(288, 384), down_ratio=4),
        #     dict(type='Normalize', **img_norm_cfg),
        #     dict(type='Pad', size_divisor=32),
        #     dict(
        #         type='DefaultFormatBundle3D',
        #         class_names=class_names,
        #         with_label=False),
        #     dict(type='Collect3D', keys=['img']),
        # ])
]
data_root_car = "input/car_test_images"
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline,
        data_root=data_root_car,
        ann_file= data_root_car + '/annotation.json',# ann_file_test,
        img_prefix=data_root_car))
