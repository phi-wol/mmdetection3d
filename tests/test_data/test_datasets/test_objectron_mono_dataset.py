# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmdet3d.datasets import ObjectronMonoDataset
from mmdet3d.models.builder import build_head

from mmdet3d.core.bbox import CameraInstance3DBoxes


def test_getitem():
    np.random.seed(0)
    class_names = ["chair"]
    img_norm_cfg = dict(
        mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    pipeline = [
        dict(type='LoadImageFromFileMono3D'),
        dict(
            type='LoadAnnotations3D',
            with_bbox=True,
            with_label=True,
            with_attr_label=False,
            with_bbox_3d=True,
            with_label_3d=True,
            with_bbox_depth=True),
        dict(type='Resize', img_scale=(720, 960), keep_ratio=True),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=1.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                'gt_labels_3d', 'centers2d', 'depths'
            ]),
    ]

    data_root = "input/objectron_processed_chair"

    kitti_dataset = ObjectronMonoDataset(
        ann_file= data_root + '/annotations/objectron_train.json',
        #info_file= data_root + '/annotations/objectron_train.json', # 'tests/data/kitti/kitti_infos_mono3d.pkl', # needed?
        pipeline=pipeline,
        data_root=data_root,
        img_prefix=data_root,
        test_mode=False)

    data = kitti_dataset[0]
    img_metas = data['img_metas']._data
    filename = img_metas['filename']
    img_shape = img_metas['img_shape']
    pad_shape = img_metas['pad_shape']
    flip = img_metas['flip']
    bboxes = data['gt_bboxes']._data
    labels3d = data['gt_labels_3d']._data
    labels = data['gt_labels']._data
    centers2d = data['centers2d']._data
    depths = data['depths']._data

    expected_filename = data_root + "/images/chair_batch-17_26_0.jpg"
    expected_img_shape = (960, 720, 3)
    expected_pad_shape = (960, 736, 3)
    expected_flip = True
    expected_bboxes = torch.tensor([[720-116-541, 140, 720-116, 140+637]]).float() # original: torch.tensor([[116, 140, 116+541, 140+637]]) # as flipped
    expected_labels = torch.tensor([0])
    expected_centers2d = torch.tensor([[720 - 329.24604892730713, 470.4561710357666]])
    expected_depths = torch.tensor([1.1857185363769531])

    print(img_shape)
    print(bboxes)
    print(labels3d)
    print(labels)
    print(centers2d)
    print(depths)

    print(data['img']._data.shape)

    assert filename == expected_filename
    assert img_shape == expected_img_shape
    assert pad_shape == expected_pad_shape
    assert flip == expected_flip

    print(bboxes)
    print(expected_bboxes)

    print(centers2d)
    print(expected_centers2d )

    assert torch.allclose(bboxes, expected_bboxes, 1e-5)
    assert torch.all(labels == expected_labels)
    assert torch.all(labels3d == expected_labels)
    assert torch.allclose(centers2d, expected_centers2d, 1e-5)
    assert torch.allclose(depths, expected_depths, 1e-5)


def test_format_results():
    root_path = 'tests/data/kitti/'
    info_file = 'tests/data/kitti/kitti_infos_mono3d.pkl'
    ann_file = 'tests/data/kitti/kitti_infos_mono3d.coco.json'
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    pipeline = [
        dict(type='LoadImageFromFileMono3D'),
        dict(
            type='LoadAnnotations3D',
            with_bbox=True,
            with_label=True,
            with_attr_label=False,
            with_bbox_3d=True,
            with_label_3d=True,
            with_bbox_depth=True),
        dict(type='Resize', img_scale=(1242, 375), keep_ratio=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                'gt_labels_3d', 'centers2d', 'depths'
            ]),
    ]
    kitti_dataset = KittiMonoDatasetObjectron(
        ann_file=ann_file,
        info_file=info_file,
        pipeline=pipeline,
        data_root=root_path,
        test_mode=True)

    # format 3D detection results
    results = mmcv.load('tests/data/kitti/mono3d_sample_results.pkl')
    result_files, tmp_dir = kitti_dataset.format_results(results)
    result_data = result_files['img_bbox']
    assert len(result_data) == 1
    assert len(result_data[0]['name']) == 4
    det = result_data[0]

    expected_bbox = torch.tensor([[565.4989, 175.02547, 616.70184, 225.00565],
                                  [481.85907, 179.8642, 512.43414, 202.5624],
                                  [542.23157, 175.73912, 565.26263, 193.96303],
                                  [330.8572, 176.1482, 355.53937, 213.8469]])
    expected_dims = torch.tensor([[3.201, 1.6110001, 1.661],
                                  [3.701, 1.401, 1.511],
                                  [4.051, 1.4610001, 1.661],
                                  [1.9510001, 1.7210001, 0.501]])
    expected_rotation = torch.tensor([-1.59, 1.55, 1.56, 1.54])
    expected_detname = ['Car', 'Car', 'Car', 'Cyclist']

    assert torch.allclose(torch.from_numpy(det['bbox']), expected_bbox, 1e-5)
    assert torch.allclose(
        torch.from_numpy(det['dimensions']), expected_dims, 1e-5)
    assert torch.allclose(
        torch.from_numpy(det['rotation_y']), expected_rotation, 1e-5)
    assert det['name'].tolist() == expected_detname

    # format 2D detection results
    results = mmcv.load('tests/data/kitti/mono3d_sample_results2d.pkl')
    result_files, tmp_dir = kitti_dataset.format_results(results)
    result_data = result_files['img_bbox2d']
    assert len(result_data) == 1
    assert len(result_data[0]['name']) == 4
    det = result_data[0]

    expected_bbox = torch.tensor(
        [[330.84191493, 176.13804312, 355.49885373, 213.81578769],
         [565.48227204, 175.01202566, 616.65650883, 224.96147091],
         [481.84967085, 179.85710612, 512.41043776, 202.54001526],
         [542.22471517, 175.73341152, 565.24534908, 193.94568878]])
    expected_dims = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                                  [0., 0., 0.]])
    expected_rotation = torch.tensor([0., 0., 0., 0.])
    expected_detname = ['Cyclist', 'Car', 'Car', 'Car']

    assert torch.allclose(
        torch.from_numpy(det['bbox']).float(), expected_bbox, 1e-5)
    assert torch.allclose(
        torch.from_numpy(det['dimensions']).float(), expected_dims, 1e-5)
    assert torch.allclose(
        torch.from_numpy(det['rotation_y']).float(), expected_rotation, 1e-5)
    assert det['name'].tolist() == expected_detname


def test_evaluate():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    root_path = "input/objectron_processed_chair"
    #info_file = 'tests/data/kitti/kitti_infos_mono3d.pkl'
    ann_file= root_path + '/annotations/objectron_train.json'
    class_names = ['chair']
    pipeline = [
        dict(type='LoadImageFromFileMono3D'),
        dict(
            type='LoadAnnotations3D',
            with_bbox=True,
            with_label=True,
            with_attr_label=False,
            with_bbox_3d=True,
            with_label_3d=True,
            with_bbox_depth=True),
        dict(type='Resize', img_scale= (720, 960), keep_ratio=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                'gt_labels_3d', 'centers2d', 'depths'
            ]),
    ]
    kitti_dataset = ObjectronMonoDataset(
        ann_file=ann_file,
        #info_file=info_file,
        pipeline=pipeline,
        data_root=root_path,
        test_mode=True)

    # format 3D detection results
    # results = mmcv.load('tests/data/kitti/mono3d_sample_results.pkl')
    # results2d = mmcv.load('tests/data/kitti/mono3d_sample_results2d.pkl')
    # results[0]['img_bbox2d'] = results2d[0]['img_bbox2d']

    
    ## objectron head
    num_classes = 1
    head_cfg = dict(
        type='SMOKEMono3DHeadObjectron',
        num_classes=num_classes,
        in_channels=64,
        dim_channel=[3, 4, 5],
        ori_channel=[6, 7, 8, 9, 10, 11],
        stacked_convs=0,
        feat_channels=64,
        use_direction_classifier=False,
        diff_rad_by_sin=False,
        pred_attrs=False,
        pred_velo=False,
        dir_offset=0,
        strides=None,
        group_reg_dims=(12, ),
        cls_branch=(256, ),
        reg_branch=((256, ), ),
        num_attrs=0,
        bbox_code_size=9,
        dir_branch=(),
        attr_branch=(),
        bbox_coder=dict(
            type='SMOKECoderObjectron', #TODO: new box coder
            base_depth=(28.01, 16.32), #Hyperparameter
            base_dims=((0.50999998, 0.8281818,  0.51636363),), # for now only dummy chair dims
            code_size=9),
        loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=1 / 300),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=None,
        conv_bias=True,
        dcn_on_last_conv=False)

    self = build_head(head_cfg)

    feats = [torch.rand([11, 64, 32, 32], dtype=torch.float32)]

    # test forward
    with torch.no_grad():
        ret_dict = self(feats)

    img_metas = [
        dict(
            cam2img=[[1260.8474446004698, 0.0, 807.968244525554, 40.1111],
                     [0.0, 1260.8474446004698, 495.3344268742088, 2.34422],
                     [0.0, 0.0, 1.0, 0.00333333], [0.0, 0.0, 0.0, 1.0]],
            scale_factor=np.array([1., 1., 1., 1.], dtype=np.float32),
            pad_shape=[128, 128],
            trans_mat=np.array([[0.25, 0., 0.], [0., 0.25, 0], [0., 0., 1.]],
                               dtype=np.float32),
            affine_aug=False,
            box_type_3d=CameraInstance3DBoxes) for i in range(11)
    ]

    # test get_boxes
    results = self.get_bboxes(*ret_dict, img_metas)
    # list[tuple[:obj:`CameraInstance3DBoxes`, Tensor, Tensor, None]]

    print(results)

    # theoretisch eine weitere nested dict Schicht mit key: ['img_bbox']
    results = [dict(boxes_3d=result[0], scores_3d=result[1], labels_3d=result[2]) for result in results]

    metric = ['mAP']
    ap_dict = kitti_dataset.evaluate(results, metric)
    print(ap_dict)
