# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import glob, os
import mmcv
from os import path as osp
from mmdet3d.apis import (direct_inference_mono_3d_detector, inference_mono_3d_detector, init_model,
                          show_result_meshlab)

class VideoFrames():
    def __init__(self, filename, downsampling=1) -> None:
        self.filename = filename
        self.capture = cv2.VideoCapture(filename)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.idx = 0
        self.downsampling = downsampling
        self.first_frame = None

    def __iter__(self):
        return self

    def __next__(self):
        success, frame = self.capture.read()
        if not success:
            raise StopIteration

        self.idx += 1

        frame = cv2.resize(frame, self.get_output_size())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame

    def get_output_size(self):
        return int(self.width * self.downsampling), int(self.height * self.downsampling)

    def frame_at(self, idx):
        if idx == 0:
            if self.first_frame is not None: 
                return self.first_frame
            else:
                self.first_frame = next(self)
                return self.first_frame

        if idx < self.idx:
            raise ValueError(f"Can only skip, cannot jump back, or re-read!, current idx: {self.idx}, desired idx: {idx}")
        
        if idx == self.idx:
            return next(self)
        
        assert idx >= self.idx

        diff = idx - self.idx
        for _ in range(diff):
            next(self)

        return next(self)

    def show_interactively(self):
        frame = next(self)
        while frame is not None:
            frame = next(self)
            cv2.imshow("", frame)
            cv2.waitKey(int(1000 / self.fps))

    def __len__(self) -> int:
        return self.frame_count

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


if __name__ == '__main__':
    main()


def video_animation_9d(mov_filename, f_x=1492, f_y=1492, h=1920, w=1440, o_x=720, o_y=960, res_divisor=1):

    # load camera intrinsics from json
    camera_intrinsics = CameraIntrinsics(f_x * res_divisor, f_y * res_divisor, h * res_divisor, w * res_divisor, o_x * res_divisor, o_y * res_divisor)
    #camera_intrinsics[:2] *= res_divisor
    #trajectory = Trajectory(csv_filename)
    video_frames = VideoFrames(mov_filename)

    output_name = mov_filename.split("/")[-1].split(".")[0]

    perception9d = Perception9D()



    tracked_detection_sequence = perception9d.track_detection_sequence_camera(video_frames, visualize=True, intrinsics=camera_intrinsics, output_name=output_name)

    args_config = './input/smoke9D_objectron_generalize_full_ds_multigpu_bike_video.py'
        args_checkpoint = './output/2022_07_18_generalize_bike_multi_full/epoch_28.pth'
            

        args_device = 'cuda:0'
        
        detector = init_model(args_config, args_checkpoint, device=args_device)

        #print(intrinsics.as_matrix_3x3())

        #detector = Detector("det")
        
        if visualize: 
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            out = cv2.VideoWriter("./viz/share/detections_" + output_name + ".mp4", fourcc, video_frames.fps, video_frames.get_output_size())
        
        frame_id = 0

        sort_per_class_id = dict()

        raw_detection_sequence = {}
        tracked_detection_sequence = {}

        mot_tracker = AB3DMOT(max_age=20, min_hits=7) 
        
        for frame in tqdm(iter(video_frames)):

            # SIMPLE FRAME-BASED PREDICTIONS

            #prediction = detector.get_prediction(frame)

            # TODO: resize image and camera intrinsics
            result, data = direct_inference_mono_3d_detector(detector, frame, intrinsics.as_matrix_3x3())

            result = result[0]['img_bbox']
            # print(result)
            # print(result['boxes_3d'].tensor)
            # print(len(result['scores_3d']))
            # print(result['labels_3d'])
            
            # instances = Instances(image_size = frame.shape, fields={
            #     'pred_boxes': result['boxes_3d'],  #.tensor
            #     'scores': result['scores_3d'], 
            #     'pred_classes': result['labels_3d']})

            # print(instances)
            # print(instances.pred_boxes)

            output = []
            for ann_id in range(len(result['scores_3d'])):
                class_type = result['labels_3d'][ann_id]
                box_2d = [-1, -1, -1, -1]
                score = result['scores_3d'][ann_id]
                box_3d = result['boxes_3d'].tensor[ann_id]
                output.append([frame_id] + [class_type.tolist()] + box_2d + [score.tolist()] + box_3d.tolist())
            
            #print(output)
            
            #result['boxes_3d'].tensor, result['scores_3d'], result['labels_3d']

            

            #{'instances': Instances(num_instances=0, image_height=216, image_width=384, fields=[pred_boxes: Boxes(tensor([], device='cuda:0', size=(0, 4))), scores: tensor([], device='cuda:0'), pred_classes: tensor([], device='cuda:0', dtype=torch.int64)])}

            #instances = instances[instances.scores > score_threshold]
            #TODO: prediction["instances"] = instances

            #results to instances

            # function results to detections
            # TODO: raw_detections = self._instances_to_detections(instances, detector.get_detection_classes_list())

            raw_detections = self._instances_to_detections(result)
            #print(raw_detections)

            # print(frame_id, detections)
            raw_detection_sequence[frame_id] = [raw_detection for raw_detection in raw_detections]                    

            # CLASS-WISE TRACKING
            # AB3DMOT

            tracked_detection_sequence[frame_id] = []

            dets = np.array(output)
            # print(dets.shape)
            # print("len", len(dets))
            # if len(dets) == 0: 
            #     print("No detections")
            #     dets = np.empty((1,16))
            #     dets[:] = np.NaN
            if len(dets.shape) == 1: dets = np.expand_dims(dets, axis=0) 

            # print("Dets: ", dets)
            # print("Dets shape: ", dets.shape)
            # print(dets.size)

            # frame, class,  x1, y1, w, h,  score
            # 0,     1,      2, 3, 4, 5,    6
            additional_info = dets[: , 1:7] # dets[:, 0] == frame		

            # x, y, z, h, w, l, alpha, beta, gamma in camera coordinate follwing mmdet convention
            # 7, 6, 8,  9, 10, 11,  12, 13, 14
            dets = dets[: , 7:16] # seq_dets[:,0] == frame
            dets_all = {'dets': dets, 'info': additional_info}

            # TODO: track in global coordinates, not camera coordinates
            trackers = mot_tracker.update(dets_all)

            tracked_detections = []
            # saving results, loop over each tracklet			
            for d in trackers:
                #print(d)
                bbox3d_tmp = d[0:9]       # h, w, l, x, y, z, theta in camera coordinate
                #print(bbox3d_tmp)
                #print(" ID : ", d[9])
                id_tmp = d[9]
                #print(id_tmp)
                # ori_tmp = d[8:11]
                #print('Class: ', d[10])
                pred_class = d[10]
                type_tmp = det_id2str[pred_class] # 11
                bbox2d_tmp_trk = d[11:15] # 16
                conf_tmp = d[15]

                # save in detection format with track ID, can be used for dection evaluation and tracking visualization
                str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp,
                    bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
                    bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], 
                    bbox3d_tmp[6], bbox3d_tmp[7], bbox3d_tmp[8], conf_tmp, id_tmp)
                print(str_to_srite)

                x, y, z, l, w, h, alpha, beta, gamma = bbox3d_tmp
                detection = Detection9D(x, y, z, l, w, h, alpha, beta, gamma, conf_tmp, type_tmp, id_tmp)
                tracked_detections.append(detection)


            # Visualization

            # load detections, N x 15
            #dets = pass

            # class_ids = np.unique(instances.pred_classes.numpy())
            # for class_id in class_ids: # probably not needed in new tracker
            #     if not class_id in sort_per_class_id:
            #         sort_per_class_id[class_id] = Sort(max_age=20, min_hits=7, iou_threshold=.2)
                
            #     sort = sort_per_class_id[class_id]
                
            #     class_detections = instances[instances.pred_classes == class_id]
            #     dets = np.concatenate([class_detections.pred_boxes.tensor.numpy(), class_detections.scores.unsqueeze(axis=1).numpy()], axis=1)
            #     dets_with_id = sort.update(dets)
            #     ids = dets_with_id[:,4].astype(np.int32)

            # tracked_instances = Instances(
            #         instances.image_size, 
            #         pred_boxes=Boxes(tensor(dets_with_id[:,0:4])), 
            #         pred_classes=tensor([class_id]*dets_with_id.shape[0]),
            #         ids=tensor(ids),
            #         scores=tensor([0]*dets_with_id.shape[0])) # the tracking algorithm unfortunately eliminated this field

            # tracked_detections = self._instances_to_detections(tracked_instances, detector.get_detection_classes_list())
                

            tracked_detection_sequence[frame_id].extend([tracked_detection for tracked_detection in tracked_detections])
                
            if visualize:
                # Visualization
                # frame = detector.visualize_prediction(frame, {
                #     "instances": tracked_instances
                # })

				#print(tensor)

                image_tmp = frame
                for det in tracked_detections:
                    tensor = det.get_tensor()
                    boxes = CameraInstance3DBoxes(np.array([tensor]))

                    if det.score < score_threshold:
                        continue

                    color_tmp = tuple([int(tmp * 255) for tmp in colors[int(det.instance_id) % max_color]])
                    

                    # TODO: add class label to plot
                    print(det.instance_id)
                    image_tmp = draw_camera_bbox3d_on_img(
                    boxes, image_tmp, intrinsics.as_matrix_3x3(), None, color=color_tmp, thickness=3, object_id=int(det.instance_id), class_type=det.class_name)

            if visualize: 
                image_tmp = cv2.cvtColor(image_tmp, cv2.COLOR_RGB2BGR)
                out.write(image_tmp)

            frame_id += 1
            #print(".", end="")

            # if frame_id > 20:
            #     break

        with open("./viz/share/raw_detections.json", "wt") as file:
            json.dump(raw_detection_sequence, file, default=_to_dict)
        
        with open("./viz/share/tracked_detections.json", "wt") as file:
            json.dump(tracked_detection_sequence, file, default=_to_dict)

        if visualize:
            out.release() 

        return tracked_detection_sequence
