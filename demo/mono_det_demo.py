# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import mmcv
from os import path as osp
import time 
from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)


def main():
    parser = ArgumentParser()
    parser.add_argument('image', help='image file')
    parser.add_argument('ann', help='ann file')
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

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    start = time.time()
    print("start", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()), start)
    result, data = inference_mono_3d_detector(model, args.image, args.ann)
    stop = time.time() 
    print("stop", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()), stop)
    print("duration", stop-start)
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
    print(result)
    # torch tensors not serial -> pkl format

    file_name = osp.split(args.image)[-1].split('.')[0]
    result_path = osp.join(args.out_dir, file_name)

    mmcv.dump(result, result_path + '/' + file_name + "result.pkl")
    # make json serializable
    save_dict = {
        'boxes_3d': result[0]["boxes_3d"].tensor.tolist(),
        'scores_3d': result[0]["scores_3d"].tolist(),
        'labels_3d':  result[0]["labels_3d"].tolist()
        }
    mmcv.dump(save_dict, result_path + '/' + file_name + "result.json")


if __name__ == '__main__':
    main()
