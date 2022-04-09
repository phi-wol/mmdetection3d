# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import glob, os
import mmcv
from os import path as osp
from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)


def main():
    parser = ArgumentParser()
    parser.add_argument('folder_path', help='image files')
    parser.add_argument('ann', help='ann file') # assume annotation file within folder
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # gather all image_paths
    # os.chdir("/mydir")
    image_list = glob.glob(args.folder_path + "/*.png")

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    save_dict = {}
    results = {}

    for image_name in image_list:

        # test a single image
        #image_path = args.folder_path + '/' + image_name
        result, data = inference_mono_3d_detector(model, image_name, args.ann)
        # show the results
        show_result_meshlab(
            data,
            result,
            args.out_dir,
            args.score_thr,
            show=args.show,
            snapshot=args.snapshot,
            task='mono-det')

        # print numerical output values
        print(image_name)
        print(result)

        results[image_name] = result[0]
        
        # make json serializable
        save_dict[image_name] = {
            'boxes_3d': result[0]["boxes_3d"].tensor.tolist(),
            'scores_3d': result[0]["scores_3d"].tolist(),
            'labels_3d':  result[0]["labels_3d"].tolist()
            }

    # torch tensors not serial -> pkl format
    # file_name = osp.split(image_name)[-1].split('.')[0]
    # result_path = osp.join(args.out_dir, file_name)

    mmcv.dump(results, args.out_dir + '/results.pkl')
    mmcv.dump(save_dict, args.out_dir + '/results.json')


def test_inference_mono_3d_detector():
    # FCOS3D only has GPU implementations
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    img = 'tests/data/nuscenes/samples/CAM_BACK_LEFT/' \
          'n015-2018-07-18-11-07-57+0800__CAM_BACK_LEFT__1531883530447423.jpg'
    ann_file = 'tests/data/nuscenes/nus_infos_mono3d.coco.json'
    detector_cfg = 'configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_' \
                   '2x8_1x_nus-mono3d.py'
    detector = init_model(detector_cfg, device='cuda:0')
    results = inference_mono_3d_detector(detector, img, ann_file)
    bboxes_3d = results[0][0]['img_bbox']['boxes_3d']
    scores_3d = results[0][0]['img_bbox']['scores_3d']
    labels_3d = results[0][0]['img_bbox']['labels_3d']


if __name__ == '__main__':
    main()
