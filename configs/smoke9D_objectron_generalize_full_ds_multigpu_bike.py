_base_ = ['../configs/_base_/default_runtime.py'
] #'../configs/_base_/models/smoke.py' already copied below

workflow = [('train', 1), ('val', 1)]
#evaluation = dict(interval=1)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
    
# testing new folders:
#data_root = "objectron_processed_chair_all"
#ann_file_test= data_root + '/annotations/objectron_test.json', # defines 'LoadImageFromFileMono3D' & 'LoadAnnotations3D'
# info_file = data_root + 'infos.pkl'
#ann_file_train = data_root + '/annotations/objectron_train_single.json',

# scale = (1920, 1440) # original resolution
# size_divisor = 2 (960, 720)
scale = (480, 640) # 4 (480, 360)   (640, 480)

# optimizer
optimizer = dict(type='Adam', lr=2.5e-5)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup=None, step=[13, 18]) # [5, 8]# 50, 100, 150

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

find_unused_parameters = True
class_names = ["bike"]# ["chair", "book"] # TODO: single class?
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #TODO: Change on images from different distribution?

dataset_type = 'ObjectronMonoDataset' # TODO: new dataset? change loading and evaluation procedure # 'KittiMonoDatasetObjectron'
input_modality = dict(use_lidar=False, use_camera=True)

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
    dict(type='AffineResize', img_scale=scale, down_ratio=4),
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
    dict(
        type='MultiScaleFlipAug',
        img_scale=scale,
        flip=False,
        transforms=[
            dict(type='AffineResize', img_scale=scale, down_ratio=4),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
data_root_chair = "objectron_processed_chair_all"
data_root_book = "objectron_processed_book_all"
data_root_bike = "objectron_processed_bike_all"
#data_root_bike = "./input/objectron_processed_bike_overfit"

train_chair = dict(
    type=dataset_type,
    data_root= data_root_chair,
    ann_file= data_root_chair + '/annotations/objectron_train.json', # '/annotations/objectron_train_single.json' '/annotations/objectron_train.json'
    pipeline=train_pipeline,
    img_prefix=data_root_chair,
    classes=class_names,
    modality=input_modality,
    test_mode=False,
    box_type_3d='Camera'
    )

train_book = dict(
    type=dataset_type,
    data_root= data_root_book,
    ann_file= data_root_book + '/annotations/objectron_train.json', # '/annotations/objectron_train_single.json' '/annotations/objectron_train.json'
    pipeline=train_pipeline,
    img_prefix=data_root_book,
    classes=class_names,
    modality=input_modality,
    test_mode=False,
    box_type_3d='Camera'
    )

val_chair = dict(
    type=dataset_type,
    pipeline=test_pipeline,
    data_root= data_root_chair,
    ann_file = data_root_chair + '/annotations/objectron_test.json',
    img_prefix= data_root_chair,
    modality=input_modality,
    test_mode=True,
    box_type_3d='Camera'
    )

val_book = dict(
    type=dataset_type,
    pipeline=test_pipeline,
    data_root= data_root_book,
    ann_file = data_root_book + '/annotations/objectron_test.json',
    img_prefix= data_root_book,
    modality=input_modality,
    test_mode=True,
    box_type_3d='Camera'
    )

test_chair=dict(
    type=dataset_type,
    data_root=data_root_chair,
    ann_file= data_root_chair + '/annotations/objectron_test.json',
    img_prefix=data_root_chair,
    classes=class_names,
    pipeline=test_pipeline, # TODO: Change?
    modality=input_modality,
    test_mode=True,
    box_type_3d='Camera'
    )

test_book=dict(
    type=dataset_type,
    data_root=data_root_book,
    ann_file = data_root_book + '/annotations/objectron_test.json',
    img_prefix= data_root_book,
    classes=class_names,
    pipeline=test_pipeline, # TODO: Change?
    modality=input_modality,
    test_mode=True,
    box_type_3d='Camera'
    )

test_inference=dict(
        type=dataset_type,
        data_root=data_root_chair,
        ann_file= data_root_chair + '/annotations/objectron_test.json',# ann_file_test,
        img_prefix=data_root_chair,
        classes=class_names,
        pipeline=test_pipeline, # TODO: Change?
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera')


train_bike = dict(
    type=dataset_type,
    data_root= data_root_bike,
    ann_file= data_root_bike + '/annotations/objectron_train.json', # '/annotations/objectron_train_single.json' '/annotations/objectron_train.json'
    pipeline=train_pipeline,
    img_prefix=data_root_bike,
    classes=class_names,
    modality=input_modality,
    test_mode=False,
    box_type_3d='Camera'
    )

val_bike = dict(
    type=dataset_type,
    pipeline=test_pipeline,
    data_root= data_root_bike,
    ann_file = data_root_bike + '/annotations/objectron_test.json',
    img_prefix= data_root_bike,
    modality=input_modality,
    test_mode=True,
    box_type_3d='Camera'
    )

test_bike=dict(
    type=dataset_type,
    data_root=data_root_bike,
    ann_file= data_root_bike + '/annotations/objectron_test.json',
    img_prefix=data_root_bike,
    classes=class_names,
    pipeline=test_pipeline, # TODO: Change?
    modality=input_modality,
    test_mode=True,
    box_type_3d='Camera'
    )

# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=4,
#     train = dict(
#         type='ConcatDataset',
#         datasets= [train_chair, train_book],
#         separate_eval=True
#     ),
#     val= dict(
#         type='ConcatDataset',
#         datasets= [val_chair, val_book],
#         separate_eval=True
#     ),
#     #test = test_inference
#     test= dict(
#         type='ConcatDataset',
#         datasets= [test_chair, test_book],
#         separate_eval=True
#     )
# )

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train = train_bike,
    val= val_bike,
    #test = test_inference
    test= test_bike,
)

model = dict(
    type='SMOKEMono3D', 
    backbone=dict(
        type='DLANet',
        depth=34,
        in_channels=3,
        norm_cfg=dict(type='GN', num_groups=32),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
        )),
    neck=dict(
        type='DLANeck',
        in_channels=[16, 32, 64, 128, 256, 512],
        start_level=2,
        end_level=5,
        norm_cfg=dict(type='GN', num_groups=32)),
    bbox_head=dict(
        type='SMOKEMono3DHeadObjectronPrint', #TODO: new head  SMOKEMono3DHeadObjectronEuler
        num_classes=2, # TODO: focus on single class overfit
        in_channels=64,
        dim_channel=[3, 4, 5],
        ori_channel=[6, 7, 8, 9, 10, 11], # added new ori channels
        stacked_convs=0,
        feat_channels=64,
        use_direction_classifier=False,
        diff_rad_by_sin=False,
        pred_attrs=False,
        pred_velo=False,
        dir_offset=0,
        strides=None,
        print_loss = False,
        print_corners = False,
        group_reg_dims=(12, ), # changed from 8 to 12: this parameter influences number of regression output channels
        cls_branch=(256, ),
        reg_branch=((256, ), ),
        num_attrs=0,
        bbox_code_size=9, # for 2 additional orientation angles
        dir_branch=(),
        attr_branch=(),
        bbox_coder=dict(
            type='SMOKECoderObjectron', #TODO: new box coder
            base_depth=(1.5, 0.5), #Hyperparameter (28.01, 16.32)
            base_dims=  (
                (0.65320896, 1.021797894, 1.519635599), # bike
                (0.5740664085137888, 0.8434027515832329, 0.6051523831888338), # chair
                (0.225618019, 0.03949624326, 0.1625821624) # book
            ), # ((2.0, 2.0, 2.0),), # ((0.50999998, 0.8281818,  0.51636363),), # for now only dummy chair dims
            code_size=9),
        loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=1.0 /300 ), #/ 300),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=None,
        conv_bias=True,
        dcn_on_last_conv=False),
    train_cfg=None,
    test_cfg=dict(topK=100, local_maximum_kernel=3, max_per_img=100))