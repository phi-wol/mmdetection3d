
#%%
# %load_ext autoreload
# %autoreload 2

#%%
# !export PYTHONPATH=${PYTHONPATH}:/Users/philipp.wolters/code/semantic_perception/AB3DMOT
# !export PYTHONPATH=${PYTHONPATH}:/Users/philipp.wolters/code/semantic_perception/AB3DMOT/Xinshuo_PyToolbox


#%%
from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import os, numpy as np, time, sys
from AB3DMOT_libs.model import AB3DMOT
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing

#%%

result_sha = 'multi' # 'custom' #'pred' #'gt' #sys.argv[1]
save_root = './results'
# det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
det_id2str = {0: 'bike', 1:'chair', 2:'book'}

seq_file_list, num_seq = load_list_from_folder(os.path.join('data/objectron', result_sha))

#print(seq_file_list, num_seq)
#%%

total_time, total_frames = 0.0, 0
save_dir = os.path.join(save_root, result_sha); mkdir_if_missing(save_dir)
eval_dir = os.path.join(save_dir, 'data'); mkdir_if_missing(eval_dir)
#print(eval_dir)
seq_count = 0
for seq_file in seq_file_list:
    _, seq_name, _ = fileparts(seq_file)
    eval_file = os.path.join(eval_dir, seq_name + '.txt'); eval_file = open(eval_file, 'w')
    save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name); mkdir_if_missing(save_trk_dir)

    mot_tracker = AB3DMOT(max_age=2, min_hits=3) # 30 5
    seq_dets = np.loadtxt(seq_file, delimiter=',')          # load detections, N x 15
    
    # if no detection in a sequence
    if len(seq_dets.shape) == 1: seq_dets = np.expand_dims(seq_dets, axis=0) 	
    if seq_dets.shape[1] == 0:
        eval_file.close()
        continue
    min_frame, max_frame = int(seq_dets[:, 0].min()), int(seq_dets[:, 0].max())

    for frame in range(min_frame, max_frame + 1):
        # logging
        print_str = 'processing %s: %d/%d, %d/%d   \r' % (seq_name, seq_count, num_seq, frame, max_frame)
        #sys.stdout.write(print_str)
        #sys.stdout.flush()
        print(print_str)
        save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame); save_trk_file = open(save_trk_file, 'w')

        # get irrelevant information associated with an object, not used for associationg
        # ori_array = seq_dets[seq_dets[:, 0] == frame, -3:].reshape((-1, 3))		# orientation
        # print(ori_array)

        # other_array = seq_dets[seq_dets[:, 0] == frame, 1:7] 		# other information, e.g, 2D box, ...
        # additional_info = np.concatenate((ori_array, other_array), axis=1)

        # frame, class,  x1, y1, w, h,  score
        # 0,     1,      2, 3, 4, 5,    6
        additional_info = seq_dets[seq_dets[:, 0] == frame, 1:7] 		

        # x, y, z, h, w, l, alpha, beta, gamma in camera coordinate follwing mmdet convention
        # 7, 6, 8,  9, 10, 11,  12, 13, 14
        dets = seq_dets[seq_dets[:,0] == frame, 7:16]           
        dets_all = {'dets': dets, 'info': additional_info}

        #print(dets_all)

        # important
        start_time = time.time()
        trackers = mot_tracker.update(dets_all)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        
        # saving results, loop over each tracklet			
        for d in trackers:
            #print(d)
            bbox3d_tmp = d[0:9]       # h, w, l, x, y, z, theta in camera coordinate
            #print(bbox3d_tmp)
            id_tmp = d[9]
            #print(id_tmp)
            # ori_tmp = d[8:11]
            #print('Class: ', d[10])
            type_tmp = det_id2str[d[10]] # 11
            bbox2d_tmp_trk = d[11:15] # 16
            conf_tmp = d[15]

            # save in detection format with track ID, can be used for dection evaluation and tracking visualization
            str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp,
                bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
                bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], 
                bbox3d_tmp[6], bbox3d_tmp[7], bbox3d_tmp[8], conf_tmp, id_tmp)
            save_trk_file.write(str_to_srite)

            # save in tracking format, for 3D MOT evaluation
            str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, 
                type_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
                bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], 
                bbox3d_tmp[6], bbox3d_tmp[7], bbox3d_tmp[8], conf_tmp)
            eval_file.write(str_to_srite)

        total_frames += 1
        save_trk_file.close()
    seq_count += 1
    eval_file.close()    
print('Total Tracking took: %.3f for %d frames or %.1f FPS' % (total_time, total_frames, total_frames / total_time))