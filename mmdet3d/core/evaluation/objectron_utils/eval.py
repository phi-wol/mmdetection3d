

# Copyright (c) OpenMMLab. All rights reserved.
import gc
import io as sysio

import numba
import numpy as np


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


# max truncated or occluded
def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno['bbox'][i]
        gt_name = gt_anno['name'][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == 'Pedestrian'.lower()
              and 'Person_sitting'.lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno['occluded'][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno['truncated'][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno['name'][i] == 'DontCare':
            dc_bboxes.append(gt_anno['bbox'][i])
    for i in range(num_dt):
        if (dt_anno['name'][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno['bbox'][i, 3] - dt_anno['bbox'][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]) + qbox_area -
                              iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps

@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i],
                               gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    pass


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]['bbox'], gt_annos[i]['alpha'][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]['bbox'], dt_annos[i]['alpha'][..., np.newaxis],
            dt_annos[i]['score'][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=200):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        current_classes (list[int]): 0: car, 1: pedestrian, 2: cyclist.
        difficultys (list[int]): Eval difficulty, 0: easy, 1: normal, 2: hard
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps (float): Min overlap. format:
            [num_overlap, metric, class].
        num_parts (int): A parameter for fast calculate algorithm

    Returns:
        dict[str, np.ndarray]: recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for idx_l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, idx_l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, idx_l, k, i] = pr[i, 0] / (
                        pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, idx_l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, idx_l, k, i] = np.max(
                        precision[m, idx_l, k, i:], axis=-1)
                    recall[m, idx_l, k, i] = np.max(
                        recall[m, idx_l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, idx_l, k, i] = np.max(
                            aos[m, idx_l, k, i:], axis=-1)
    ret_dict = {
        'recall': recall,
        'precision': precision,
        'orientation': aos,
    }

    # clean temp variables
    del overlaps
    del parted_overlaps

    gc.collect()
    return ret_dict


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            eval_types=['bbox', 'bev', '3d']):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]
    mAP11_bbox = None
    mAP11_aos = None
    mAP40_bbox = None
    mAP40_aos = None
    if 'bbox' in eval_types:
        ret = eval_class(
            gt_annos,
            dt_annos,
            current_classes,
            difficultys,
            0,
            min_overlaps,
            compute_aos=('aos' in eval_types))
        # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
        mAP11_bbox = get_mAP11(ret['precision'])
        mAP40_bbox = get_mAP40(ret['precision'])
        if 'aos' in eval_types:
            mAP11_aos = get_mAP11(ret['orientation'])
            mAP40_aos = get_mAP40(ret['orientation'])

    mAP11_bev = None
    mAP40_bev = None
    if 'bev' in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                         min_overlaps)
        mAP11_bev = get_mAP11(ret['precision'])
        mAP40_bev = get_mAP40(ret['precision'])

    mAP11_3d = None
    mAP40_3d = None
    if '3d' in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                         min_overlaps)
        mAP11_3d = get_mAP11(ret['precision'])
        mAP40_3d = get_mAP40(ret['precision'])
    return (mAP11_bbox, mAP11_bev, mAP11_3d, mAP11_aos, mAP40_bbox, mAP40_bev,
            mAP40_3d, mAP40_aos)


def objectron_eval(gt_annos,
               dt_annos,
               current_classes,
               eval_types=['bbox', 'bev', '3d']):
    """KITTI evaluation.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval.
            Defaults to ['bbox', 'bev', '3d'].

    Returns:
        tuple: String and dict of evaluation results.
    """
    assert len(eval_types) > 0, 'must contain at least one evaluation type'
    if 'aos' in eval_types:
        assert 'bbox' in eval_types, 'must evaluate bbox when evaluating aos'
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5], [0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.7, 0.5, 0.5, 0.7, 0.5]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25],
                            [0.5, 0.25, 0.25, 0.5, 0.25]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    pred_alpha = False
    valid_alpha_gt = False
    for anno in dt_annos:
        mask = (anno['alpha'] != -10)
        if anno['alpha'][mask].shape[0] != 0:
            pred_alpha = True
            break
    for anno in gt_annos:
        if anno['alpha'][0] != -10:
            valid_alpha_gt = True
            break
    compute_aos = (pred_alpha and valid_alpha_gt)
    if compute_aos:
        eval_types.append('aos')

    mAP11_bbox, mAP11_bev, mAP11_3d, mAP11_aos, mAP40_bbox, mAP40_bev, \
        mAP40_3d, mAP40_aos = do_eval(gt_annos, dt_annos,
                                      current_classes, min_overlaps,
                                      eval_types)

    ret_dict = {}
    difficulty = ['easy', 'moderate', 'hard']

    # calculate AP11
    result += '\n----------- AP11 Results ------------\n\n'
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP11@{:.2f}, {:.2f}, {:.2f}:\n'.format(
                curcls_name, *min_overlaps[i, :, j]))
            if mAP11_bbox is not None:
                result += 'bbox AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP11_bbox[j, :, i])
            if mAP11_bev is not None:
                result += 'bev  AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP11_bev[j, :, i])
            if mAP11_3d is not None:
                result += '3d   AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP11_3d[j, :, i])
            if compute_aos:
                result += 'aos  AP11:{:.2f}, {:.2f}, {:.2f}\n'.format(
                    *mAP11_aos[j, :, i])

            # prepare results for logger
            for idx in range(3):
                if i == 0:
                    postfix = f'{difficulty[idx]}_strict'
                else:
                    postfix = f'{difficulty[idx]}_loose'
                prefix = f'KITTI/{curcls_name}'
                if mAP11_3d is not None:
                    ret_dict[f'{prefix}_3D_AP11_{postfix}'] =\
                        mAP11_3d[j, idx, i]
                if mAP11_bev is not None:
                    ret_dict[f'{prefix}_BEV_AP11_{postfix}'] =\
                        mAP11_bev[j, idx, i]
                if mAP11_bbox is not None:
                    ret_dict[f'{prefix}_2D_AP11_{postfix}'] =\
                        mAP11_bbox[j, idx, i]

    # calculate mAP11 over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += ('\nOverall AP11@{}, {}, {}:\n'.format(*difficulty))
        if mAP11_bbox is not None:
            mAP11_bbox = mAP11_bbox.mean(axis=0)
            result += 'bbox AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(
                *mAP11_bbox[:, 0])
        if mAP11_bev is not None:
            mAP11_bev = mAP11_bev.mean(axis=0)
            result += 'bev  AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(
                *mAP11_bev[:, 0])
        if mAP11_3d is not None:
            mAP11_3d = mAP11_3d.mean(axis=0)
            result += '3d   AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP11_3d[:,
                                                                            0])
        if compute_aos:
            mAP11_aos = mAP11_aos.mean(axis=0)
            result += 'aos  AP11:{:.2f}, {:.2f}, {:.2f}\n'.format(
                *mAP11_aos[:, 0])

        # prepare results for logger
        for idx in range(3):
            postfix = f'{difficulty[idx]}'
            if mAP11_3d is not None:
                ret_dict[f'KITTI/Overall_3D_AP11_{postfix}'] = mAP11_3d[idx, 0]
            if mAP11_bev is not None:
                ret_dict[f'KITTI/Overall_BEV_AP11_{postfix}'] =\
                    mAP11_bev[idx, 0]
            if mAP11_bbox is not None:
                ret_dict[f'KITTI/Overall_2D_AP11_{postfix}'] =\
                    mAP11_bbox[idx, 0]

    # Calculate AP40
    result += '\n----------- AP40 Results ------------\n\n'
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP40@{:.2f}, {:.2f}, {:.2f}:\n'.format(
                curcls_name, *min_overlaps[i, :, j]))
            if mAP40_bbox is not None:
                result += 'bbox AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP40_bbox[j, :, i])
            if mAP40_bev is not None:
                result += 'bev  AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP40_bev[j, :, i])
            if mAP40_3d is not None:
                result += '3d   AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP40_3d[j, :, i])
            if compute_aos:
                result += 'aos  AP40:{:.2f}, {:.2f}, {:.2f}\n'.format(
                    *mAP40_aos[j, :, i])

            # prepare results for logger
            for idx in range(3):
                if i == 0:
                    postfix = f'{difficulty[idx]}_strict'
                else:
                    postfix = f'{difficulty[idx]}_loose'
                prefix = f'KITTI/{curcls_name}'
                if mAP40_3d is not None:
                    ret_dict[f'{prefix}_3D_AP40_{postfix}'] =\
                        mAP40_3d[j, idx, i]
                if mAP40_bev is not None:
                    ret_dict[f'{prefix}_BEV_AP40_{postfix}'] =\
                        mAP40_bev[j, idx, i]
                if mAP40_bbox is not None:
                    ret_dict[f'{prefix}_2D_AP40_{postfix}'] =\
                        mAP40_bbox[j, idx, i]

    # calculate mAP40 over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += ('\nOverall AP40@{}, {}, {}:\n'.format(*difficulty))
        if mAP40_bbox is not None:
            mAP40_bbox = mAP40_bbox.mean(axis=0)
            result += 'bbox AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                *mAP40_bbox[:, 0])
        if mAP40_bev is not None:
            mAP40_bev = mAP40_bev.mean(axis=0)
            result += 'bev  AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                *mAP40_bev[:, 0])
        if mAP40_3d is not None:
            mAP40_3d = mAP40_3d.mean(axis=0)
            result += '3d   AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP40_3d[:,
                                                                            0])
        if compute_aos:
            mAP40_aos = mAP40_aos.mean(axis=0)
            result += 'aos  AP40:{:.2f}, {:.2f}, {:.2f}\n'.format(
                *mAP40_aos[:, 0])

        # prepare results for logger
        for idx in range(3):
            postfix = f'{difficulty[idx]}'
            if mAP40_3d is not None:
                ret_dict[f'KITTI/Overall_3D_AP40_{postfix}'] = mAP40_3d[idx, 0]
            if mAP40_bev is not None:
                ret_dict[f'KITTI/Overall_BEV_AP40_{postfix}'] =\
                    mAP40_bev[idx, 0]
            if mAP40_bbox is not None:
                ret_dict[f'KITTI/Overall_2D_AP40_{postfix}'] =\
                    mAP40_bbox[idx, 0]

    return result, ret_dict
