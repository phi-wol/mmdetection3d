# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, numpy as np, sys, cv2
from PIL import Image
from xinshuo_io import is_path_exists, mkdir_if_missing, load_list_from_folder, fileparts
from xinshuo_visualization import random_colors
from xinshuo_video import generate_video_from_list
from AB3DMOT_libs.kitti_utils import read_label, compute_box_3d, draw_projected_box3d, Calibration
import mmcv
from AB3DMOT_libs.mmdet.utils import draw_camera_bbox3d_on_img, points_cam2img
from AB3DMOT_libs.mmdet.cam_box3d import CameraInstance3DBoxes

max_color = 30
colors = random_colors(max_color)       # Generate random colors
type_whitelist = ['bike', 'chair', 'book']
score_threshold = -10000
width = 720
height = 960



#data_info = mmcv.load('objectron_input/sequence_white.json')
#data_info = mmcv.load('objectron_input/two_chairs.json')
# data_info['images'][i]["cam_intrinsic"]

def vis(result_sha, data_root, result_root):
	# def show_image_with_boxes(img, objects_res, object_gt, calib, save_path, height_threshold=0):
	# 	img2 = np.copy(img) 

	# 	for obj in objects_res:

	# 		box3d_pts_2d, _ = compute_box_3d(obj, calib)
	# 		color_tmp = tuple([int(tmp * 255) for tmp in colors[obj.id % max_color]])
	# 		img2 = draw_projected_box3d(img2, box3d_pts_2d, color=color_tmp)
	# 		text = 'ID: %d' % obj.id
	# 		if box3d_pts_2d is not None:
	# 			img2 = cv2.putText(img2, text, (int(box3d_pts_2d[4, 0]), int(box3d_pts_2d[4, 1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color_tmp) 

	# 	img = Image.fromarray(img2)
	# 	img = img.resize((width, height))
	# 	img.save(save_path)
	
	for seq in seq_list:
		print(seq)
		image_dir = os.path.join("./objectron_input/", seq)
		
		#calib_file = os.path.join(data_root, 'calib/%s.txt' % seq)
		info_file = os.path.join("./objectron_input/", seq + ".json")
		data_info = mmcv.load(info_file)
		result_dir = os.path.join(result_root, '%s/trk_withid/%s' % (result_sha, seq))
		#print(data_info)
		save_3d_bbox_dir = os.path.join(result_dir, '../../trk_image_vis/%s' % seq); mkdir_if_missing(save_3d_bbox_dir)

		# load the list
		#print(image_dir)
		images_list, num_images = load_list_from_folder(image_dir)
		#print(images_list)
		print('number of images to visualize is %d' % num_images)
		start_count = 0
		for count in range(start_count, num_images):
			image_tmp = images_list[count]
			if not is_path_exists(image_tmp): 
				count += 1
				continue
			#image_index = int(fileparts(image_tmp)[1])
			image_tmp = np.array(Image.open(image_tmp))
			img_height, img_width, img_channel = image_tmp.shape

			print(result_dir)
			result_tmp = os.path.join(result_dir, '%06d.txt'%count)		# load the result
			# if not is_path_exists(result_tmp): object_res = []
			# else: object_res = read_label(result_tmp)
			if not is_path_exists(result_tmp): 
				lines = []
			else: 
				lines = [line.rstrip() for line in open(result_tmp)]
			#print(lines)

			calib_tmp = np.array(data_info['images'][count]["cam_intrinsic"])

			num_instances=0
			for object in lines:
				data = object.split(' ')
				data[1:] = [float(x) for x in data[1:]]

				# # extract label, truncation, occlusion
				type = data[0] # 'book', 'chair', ...
				# self.truncation = data[1] # truncated pixel ratio [0..1]
				# self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
				# self.alpha = data[3] # object observation angle [-pi..pi]

				# # extract 2d bounding box in 0-based coordinates
				# self.xmin = data[4] # left
				# self.ymin = data[5] # top
				# self.xmax = data[6] # right
				# self.ymax = data[7] # bottom
				# self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
				
				# extract 3d bounding box information
				tensor = np.array([[
					float(data[7]), float(data[8]), float(data[9]),
					float(data[10]),float(data[11]),float(data[12]),
					float(data[13]),float(data[14]),float(data[15])]])
				boxes = CameraInstance3DBoxes(tensor)
				score = float(data[16])
				#print(tensor)

				if score < score_threshold: continue
				if type not in type_whitelist: continue
				num_instances+=1

				id = int(data[17])
				color_tmp = tuple([int(tmp * 255) for tmp in colors[id % max_color]])

				image_tmp = draw_camera_bbox3d_on_img(
                boxes, image_tmp, calib_tmp, None, color=color_tmp, thickness=3, object_id=id)

			print('processing index: %06d, %d/%d, results from %s' % (count, count+1, num_images, result_tmp))
			#calib_tmp = Calibration(calib_file)			# load the calibration

			save_image_with_3dbbox_gt_path = os.path.join(save_3d_bbox_dir, '%06d.jpg' % (count))

			img = Image.fromarray(image_tmp)
			#img = img.resize((width, height))
			img.save(save_image_with_3dbbox_gt_path)
			#print('number of objects to plot is %d' % (num_instances))
			count += 1

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('Usage: python visualization.py result_sha(e.g., pointrcnn_Car_test_thres)')
		sys.exit(1)

	result_root = './results'
	result_sha = sys.argv[1]
	if 'gt' in result_sha: 
		data_root = './data/objectron/gt'
		seq_list = [
			# 'sequence_orange', 
			# 'sequence_white',
			'chair_batch_36_17',
			'chair_batch_36_29'
			]

	elif 'pred' in result_sha: 
		data_root = './data/objectron/pred'
		seq_list = [
			# 'chair_orange',
			# 'chair_white',
			# 'chair_batch_36_17',
			'chair_batch_36_29',
			'chair_batch_4_30',
			'chair_batch_4_02',
			# 'book_batch_47_35',
			# 'book_batch_34_44',
			# 'book_batch_31_24',
			# 'book_batch_23_24',
		]

	elif 'multi' in result_sha: 
		data_root = './data/objectron/multi'
		seq_list = [
			# 'chair_orange',
			# 'chair_white',
			# 'chair_batch_36_17',
			#'chair_batch_36_29',
			'chair_batch_4_30',
			'chair_batch_4_02',
			# 'book_batch_47_35',
			# 'book_batch_34_44',
			# 'book_batch_31_24',
			# 'book_batch_23_24',
		]

	elif 'custom' in result_sha: 
		data_root = './data/objectron/custom'
		seq_list = ['single_chair', 'single_chair_single_book', 'two_chairs_two_books', 'two_chairs']

	elif 'filtered' in result_sha: 
		data_root = './data/objectron/filtered'
		seq_list = ['single_chair_single_book_filtered']

	else:
		print("wrong split!")
		sys.exit(1)

	vis(result_sha, data_root, result_root)

	#generate_video_from_list(image_list, save_path, framerate=30, downsample=1, display=True, warning=True, debug=True)