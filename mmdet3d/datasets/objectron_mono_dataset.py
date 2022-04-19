# Copyright (c) OpenMMLab. All rights reserved.
import copy
import tempfile
import warnings
from os import path as osp

import mmcv
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from mmdet.datasets import DATASETS, CocoDataset
from ..core import show_multi_modality_result
from ..core.bbox import CameraInstance3DBoxes, get_box_type, points_cam2img
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline
from mmdet3d.core.evaluation.objectron_utils.eval_original import Evaluator
import tqdm
from mmcv.utils import print_log


@DATASETS.register_module()
class ObjectronMonoDataset(CocoDataset):
    r"""Monocular 3D detection on Objectron Dataset.

    This class serves as the API for experiments on the Objectron Dataset.

    Please refer to `Objectron Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        data_root (str): Path of dataset root.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Camera' in this class. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        eval_version (str, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        version (str, optional): Dataset version. Defaults to 'v1.0-trainval'.
    """
    # CLASSES = ['bike', 'book', 'bottle', 'cereal_box', 'camera', 'chair', 'cup', 'laptop', 'shoe']
    CLASSES = ['book', 'chair']
    # DefaultAttribute = {
    #     'car': 'vehicle.parked',
    #     'pedestrian': 'pedestrian.moving',
    #     'trailer': 'vehicle.parked',
    #     'truck': 'vehicle.parked',
    #     'bus': 'vehicle.moving',
    #     'motorcycle': 'cycle.without_rider',
    #     'construction_vehicle': 'vehicle.parked',
    #     'bicycle': 'cycle.without_rider',
    #     'barrier': '',
    #     'traffic_cone': '',
    # }
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    def __init__(self,
                 data_root,
                 load_interval=1,
                 with_velocity=False,
                 modality=None,
                 box_type_3d='Camera',
                 eval_version=None,
                 use_valid_flag=False,
                 version=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.modality = modality
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.eval_version = eval_version
        self.use_valid_flag = use_valid_flag
        self.bbox_code_size = 9
        self.version = version
        if self.eval_version is not None:
            from nuscenes.eval.detection.config import config_factory
            self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False)
        #self.objectron_evaluator = Evaluator()

        self.name2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        print(self.name2label)

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        for i, ann in enumerate(ann_info): # for every instance in the image
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_name'] not in self.CLASSES:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                # gt_labels.append(self.cat2label[ann['category_id']])
                gt_labels.append(self.name2label[ann['category_name']])
                gt_masks_ann.append(ann.get('segmentation', None))
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(-1, )
                gt_bboxes_cam3d.append(bbox_cam3d)
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_labels_3d=gt_labels_3d,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    # def get_attr_name(self, attr_idx, label_name):
    #     """Get attribute from predicted index.

    #     This is a workaround to predict attribute when the predicted velocity
    #     is not reliable. We map the predicted attribute index to the one
    #     in the attribute set. If it is consistent with the category, we will
    #     keep it. Otherwise, we will use the default attribute.

    #     Args:
    #         attr_idx (int): Attribute index.
    #         label_name (str): Predicted category name.

    #     Returns:
    #         str: Predicted attribute name.
    #     """
    #     # TODO: Simplify the variable name
    #     AttrMapping_rev2 = [
    #         'cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving',
    #         'pedestrian.standing', 'pedestrian.sitting_lying_down',
    #         'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None'
    #     ]
    #     if label_name == 'car' or label_name == 'bus' \
    #         or label_name == 'truck' or label_name == 'trailer' \
    #             or label_name == 'construction_vehicle':
    #         if AttrMapping_rev2[attr_idx] == 'vehicle.moving' or \
    #             AttrMapping_rev2[attr_idx] == 'vehicle.parked' or \
    #                 AttrMapping_rev2[attr_idx] == 'vehicle.stopped':
    #             return AttrMapping_rev2[attr_idx]
    #         else:
    #             return NuScenesMonoDataset.DefaultAttribute[label_name]
    #     elif label_name == 'pedestrian':
    #         if AttrMapping_rev2[attr_idx] == 'pedestrian.moving' or \
    #             AttrMapping_rev2[attr_idx] == 'pedestrian.standing' or \
    #                 AttrMapping_rev2[attr_idx] == \
    #                 'pedestrian.sitting_lying_down':
    #             return AttrMapping_rev2[attr_idx]
    #         else:
    #             return NuScenesMonoDataset.DefaultAttribute[label_name]
    #     elif label_name == 'bicycle' or label_name == 'motorcycle':
    #         if AttrMapping_rev2[attr_idx] == 'cycle.with_rider' or \
    #                 AttrMapping_rev2[attr_idx] == 'cycle.without_rider':
    #             return AttrMapping_rev2[attr_idx]
    #         else:
    #             return NuScenesMonoDataset.DefaultAttribute[label_name]
    #     else:
    #         return NuScenesMonoDataset.DefaultAttribute[label_name]

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            list[dict]: A list of dictionaries with the json format.
        """
        result_annos = []
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            
            img_info = self.data_infos[sample_id]
            ann_info = self.get_ann_info(sample_id)

            image_id = img_info['id'] # image_id
            
            # print(det)
            det = det['img_bbox']

            # dict: bbox, box3d_camera, scores, label_preds, sample_idx
            box_dict = self.convert_valid_bboxes(det, img_info)

            result_annos.append(box_dict)

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_objectron.json')
        print('Results writes to', res_path)
        mmcv.dump(result_annos, res_path)
        return result_annos

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='img_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'img_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=True)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
            
        result_annos = self._format_bbox(results, jsonfile_prefix)

        return result_annos, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['img_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in Objectron protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            result_names (list[str], optional): Result names in the
                metric prefix. Default: ['img_bbox'].
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        # format of output predictions results: 
        # list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)

        # enable to save results to a json file
        #result_annos, tmp_dir = self.format_results(results, jsonfile_prefix)

        result_annos = []
        mapped_class_names = self.CLASSES

        print('Start Objectron Evaluation...')
        
        # flags.mark_flag_as_required('report_file')
        # flags.mark_flag_as_required('eval_data')
        info = self.data_infos[0] 
        self.objectron_evaluator = Evaluator(height=info['height'], width=info['width'])

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            
            img_info = self.data_infos[sample_id]
            ann_info = self.get_ann_info(sample_id)

            #image_id = img_info['image_id']

            # dict: bbox, box3d_camera, scores, label_preds, sample_idx
            #box_dict = self.convert_valid_bboxes(det, img_info)

            #result_annos.append(box_dict)

            # original evaluator.evaluate(batch)
            # to numpy !
            det = det['img_bbox'] #TODO: change to result_names
            self.objectron_evaluator.evaluate_single(det, ann_info, img_info)
            # det is a dict: bbox, box3d_camera, scores, label_preds, sample_idx
            # ann_info
                    # ann = dict(
                    # bboxes=gt_bboxes,
                    # labels=gt_labels,
                    # gt_bboxes_3d=gt_bboxes_cam3d,
                    # gt_labels_3d=gt_labels_3d,
                    # centers2d=centers2d,
                    # depths=depths,
                    # bboxes_ignore=gt_bboxes_ignore,
                    # masks=gt_masks_ann,
                    # seg_map=seg_map)
            # img_info: intrinsics


        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, pipeline=pipeline)

        self.objectron_evaluator.finalize()
        result_text = self.objectron_evaluator.write_report()

        print_log(result_text, logger=logger)

        result_dict = self.objectron_evaluator.get_result_dict()

        return result_dict

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        img_info = self.data_infos[index]
        input_dict = dict(img_info=img_info)

        if load_annos:
            ann_info = self.get_ann_info(index)
            input_dict.update(dict(ann_info=ann_info))

        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]

        return data

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'img_bbox' in result.keys():
                result = result['img_bbox']
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            img, img_metas = self._extract_data(i, pipeline,
                                                ['img', 'img_metas'])
            # need to transpose channel to first dim
            img = img.numpy().transpose(1, 2, 0)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            pred_bboxes = result['boxes_3d']
            show_multi_modality_result(
                img,
                gt_bboxes,
                pred_bboxes,
                img_metas['cam2img'],
                out_dir,
                file_name,
                box_mode='camera',
                show=show)

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.
                - boxes_3d (:obj:`CameraInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.
                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        box_preds_camera = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['id'] # image_id
        intrinsics = info['cam_intrinsic']
        img_shape = (info['height'], info['width']) # TODO: double check order!

        if len(box_preds_camera) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 9]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P2 = info['calib']['P2'].astype(np.float32)
        # img_shape = info['image']['image_shape']
        # P2 = box_preds.tensor.new_tensor(P2)

        # box_preds_camera = box_preds
        # box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR,
        #                                        np.linalg.inv(rect @ Trv2c))

        box_corners = box_preds_camera.corners
        #camera_intrinsics = 
        box_corners_in_image = points_cam2img(box_corners, intrinsics)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds_camera.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                            (box_2d_preds[:, 1] < image_shape[0]) &
                            (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        valid_inds = valid_cam_inds

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 9]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

def ouput_to_objectron_obx(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
            - attrs_3d (torch.Tensor, optional): Predicted attributes.

    Returns:
        list[:obj:`ObjectronBox`]: List of standard ObjectronBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
            - attrs_3d (torch.Tensor, optional): Predicted attributes.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    attrs = None
    if 'attrs_3d' in detection:
        attrs = detection['attrs_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # convert the dim/rot to nuscbox convention
    box_dims[:, [0, 1, 2]] = box_dims[:, [2, 0, 1]]
    box_yaw = -box_yaw

    box_list = []
    for i in range(len(box3d)):
        q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        quat = q2 * q1
        velocity = (box3d.tensor[i, 7], 0.0, box3d.tensor[i, 8])
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list, attrs


def cam_nusc_box_to_global(info,
                           boxes,
                           attrs,
                           classes,
                           eval_configs,
                           eval_version='detection_cvpr_2019'):
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    attr_list = []
    for (box, attr) in zip(boxes, attrs):
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['cam2ego_rotation']))
        box.translate(np.array(info['cam2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
        attr_list.append(attr)
    return box_list, attr_list


def global_nusc_box_to_cam(info,
                           boxes,
                           classes,
                           eval_configs,
                           eval_version='detection_cvpr_2019'):
    """Convert the box from global to camera coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.translate(-np.array(info['ego2global_translation']))
        box.rotate(
            pyquaternion.Quaternion(info['ego2global_rotation']).inverse)
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to camera coord system
        box.translate(-np.array(info['cam2ego_translation']))
        box.rotate(pyquaternion.Quaternion(info['cam2ego_rotation']).inverse)
        box_list.append(box)
    return box_list


def nusc_box_to_cam_box3d(boxes):
    """Convert boxes from :obj:`NuScenesBox` to :obj:`CameraInstance3DBoxes`.

    Args:
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.

    Returns:
        tuple (:obj:`CameraInstance3DBoxes` | torch.Tensor | torch.Tensor):
            Converted 3D bounding boxes, scores and labels.
    """
    locs = torch.Tensor([b.center for b in boxes]).view(-1, 3)
    dims = torch.Tensor([b.wlh for b in boxes]).view(-1, 3)
    rots = torch.Tensor([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).view(-1, 1)
    velocity = torch.Tensor([b.velocity[0::2] for b in boxes]).view(-1, 2)

    # convert nusbox to cambox convention
    dims[:, [0, 1, 2]] = dims[:, [1, 2, 0]]
    rots = -rots

    boxes_3d = torch.cat([locs, dims, rots, velocity], dim=1).cuda()
    cam_boxes3d = CameraInstance3DBoxes(
        boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
    scores = torch.Tensor([b.score for b in boxes]).cuda()
    labels = torch.LongTensor([b.label for b in boxes]).cuda()
    nms_scores = scores.new_zeros(scores.shape[0], 10 + 1)
    indices = labels.new_tensor(list(range(scores.shape[0])))
    nms_scores[indices, labels] = scores
    return cam_boxes3d, nms_scores, labels


# def get_ann_info(self, idx):
#     """Get COCO annotation by index.
#     Args:
#         idx (int): Index of data.
#     Returns:
#         dict: Annotation info of specified index.
#     """

#     img_id = self.data_infos[idx]['id']
#     ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
#     ann_info = self.coco.load_anns(ann_ids)
#     return self._parse_ann_info(self.data_infos[idx], ann_info)


# in CustomDataset init: self.data_infos = self.load_annotations(local_path)


# def load_annotations(self, ann_file):
#     """Load annotation from COCO style annotation file.
#     Args:
#         ann_file (str): Path of annotation file.
#     Returns:
#         list[dict]: Annotation info from COCO api.
#     """

#     self.coco = COCO(ann_file)
#     # The order of returned `cat_ids` will not
#     # change with the order of the CLASSES
#     self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

#     self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
#     self.img_ids = self.coco.get_img_ids()
#     data_infos = []
#     total_ann_ids = []
#     for i in self.img_ids:
#         info = self.coco.load_imgs([i])[0]
#         info['filename'] = info['file_name']
#         data_infos.append(info)
#         ann_ids = self.coco.get_ann_ids(img_ids=[i])
#         total_ann_ids.extend(ann_ids)
#     assert len(set(total_ann_ids)) == len(
#         total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
#     return data_infos


# def getImgIds(self, imgIds=[], catIds=[]):
#         '''
#         Get img ids that satisfy given filter conditions.
#         :param imgIds (int array) : get imgs for given ids
#         :param catIds (int array) : get imgs with all given cats
#         :return: ids (int array)  : integer array of img ids
#         '''
#         imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
#         catIds = catIds if _isArrayLike(catIds) else [catIds]

#         if len(imgIds) == len(catIds) == 0:
#             ids = self.imgs.keys()
#         else:
#             ids = set(imgIds)
#             for i, catId in enumerate(catIds):
#                 if i == 0 and len(ids) == 0:
#                     ids = set(self.catToImgs[catId])
#                 else:
#                     ids &= set(self.catToImgs[catId])
#         return list(ids)


# def createIndex(self):
#     # create index
#     print('creating index...')
#     anns, cats, imgs = {}, {}, {}
#     imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
#     if 'annotations' in self.dataset:
#         for ann in self.dataset['annotations']:
#             imgToAnns[ann['image_id']].append(ann)
#             anns[ann['id']] = ann

#     if 'images' in self.dataset:
#         for img in self.dataset['images']:
#             imgs[img['id']] = img

#     if 'categories' in self.dataset:
#         for cat in self.dataset['categories']:
#             cats[cat['id']] = cat

#     if 'annotations' in self.dataset and 'categories' in self.dataset:
#         for ann in self.dataset['annotations']:
#             catToImgs[ann['category_id']].append(ann['image_id'])

#     print('index created!')

#     # create class members
#     self.anns = anns
#     self.imgToAnns = imgToAnns
#     self.catToImgs = catToImgs
#     self.imgs = imgs
#     self.cats = cats

# Coco Dataset
# def get_ann_info(self, idx):
#     """Get COCO annotation by index.
#     Args:
#         idx (int): Index of data.
#     Returns:
#         dict: Annotation info of specified index.
#     """

#     img_id = self.data_infos[idx]['id']
#     ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
#     ann_info = self.coco.load_anns(ann_ids)
#     return self._parse_ann_info(self.data_infos[idx], ann_info)