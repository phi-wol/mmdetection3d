from argparse import ArgumentParser
import mmcv
from os import path as osp
import time 
from mmdet3d.apis import (init_model, show_result_meshlab)
import re
from copy import deepcopy
from os import path as osp
import time

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes,
                          show_multi_modality_result, show_result,
                          show_seg_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
import coremltools as ct

class WrappedMMDetModel(torch.nn.Module):

    def __init__(self, model):
        super(WrappedMMDetModel, self).__init__()
        self.model = model.eval()
    
    def get_input(self, data):
        self.data = data
        self.img_metas = data['img_metas']
        
    def forward(self, x):
        res = self.model(return_loss=False, rescale=True, img_metas = self.img_metas, img = [x])[0]['img_bbox']
        # Extract the tensor we want from the output dictionary
        print("Traced result: ")
        print(res)
        x = (res['boxes_3d'].tensor, res['scores_3d'], res['labels_3d'])
        return x

def inference_mono_3d_detector(model, image, ann_file):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    data_infos = mmcv.load(ann_file)
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()), "ann_loaded")
    # find the info corresponding to this image
    for x in data_infos['images']:
        if osp.basename(x['file_name']) != osp.basename(image):
            continue
        img_info = x
        break
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()), "data found")
    data = dict(
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(cam_intrinsic=img_info['cam_intrinsic']))

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    print('After collate: ')
    print(data)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data
    print(data)
    print(model)
    print(type(model))
    # forward the model
    with torch.no_grad():
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()), "nograd")
        start = time.time()
        result = model(return_loss=False, rescale=True, **data)
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()), "result retrieved, duration: ", time.time() - start)
    return result, data

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
    
    #######


    model.to(torch.device('cpu'))
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

    print('Starting model export ...')

    to_core_ml(model, data)

def to_onnx(model, example_input):
    #model  needs to be in eval mode
    model.eval()

    model_wrapped = WrappedMMDetModel(model)
    model_wrapped.get_input(example_input)
    tensor = example_input['img'][0]

    input_names = [ "actual_input" ]
    output_names = [ "output" ]

    torch.onnx.export(model_wrapped,
        tensor,
        "SMOKE9D.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )

def to_core_ml(model, example_input):

    #generate random tensor and trace the model 
    example_input2 = torch.rand(1,1,28,28)
    model_wrapped = WrappedMMDetModel(model)
    model_wrapped.get_input(example_input)
    tensor = example_input['img'][0]
    print(tensor.shape)
    traced_model = torch.jit.trace(model_wrapped, tensor)
    traced_model(tensor) # not sure if this is needed explicitly found in (https://coremltools.readme.io/docs/pytorch-conversion)

    #define class labels and config to be used by the coreML model
    class_labels = ['chair', 'book']

    testModel = ct.convert(
       traced_model,
       inputs=[ct.TensorType(shape=tensor.shape)]
    )
    classifier_config = ct.ClassifierConfig(class_labels)

    #define input name 
    input_image = ct.ImageType(name="my_input", shape=(1, 1, 28, 28), scale=1/255)
    
    #convert pytorch model to coreMLmodel
    testModel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=tensor.shape)],
        classifier_config=classifier_config
    )

    # OPTIONAL add descriptions for better readability
    testModel.input_description["my_input"] = "Input image to be classified"
    testModel.output_description["classLabel"] = "Most likely image category"

    save model
    testModel.save("SMOKE9D.mlmodel")

if __name__ == '__main__':
    main()
