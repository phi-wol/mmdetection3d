def extract_keyframes(input_file='./output3/20220819_091024-car_multi_person_7-scene_frames.json', fps=60, bbox_frame_index=500):
        # MuCo joint set
        joint_num = 21
        joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        joints_name_dict = {}
        for i, name in enumerate(joints_name):
            joints_name_dict[name] = i

        pelvs_idx = 14

        print(joints_name[14])
        print(joints_name_dict['Pelvis'])

        flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
        
        with open(input_file, "rt") as file:
            scene_frames_dict = json.load(file)

        frame_index = 0
        indices = list(scene_frames_dict.keys())
        #print(scene_frames_dict.keys())
        #print(scene_frames_dict[str(bbox_frame_index)])


        # 3D bounding boxes assumed static
        human_cars_dict = scene_frames_dict[str(bbox_frame_index)]

        pred_objects = human_cars_dict["objects"]

        print("Number of oriented bboxes: ", len(pred_objects))

        # humans need a for loop: 
        pred_humans = human_cars_dict["humans"] # reference frame 

        keypoint_ids, keypoint_list = zip(*pred_humans.items()) # TODO: also include human ids, which were not present in the selected frame

        # get list of human keypoints over all frames

        keypoints_along_frames = []

        # downsampling_rate = 10 
        # [::downsampling_rate]

        for index in indices:
            keypoints_along_frames.append(scene_frames_dict[index]["humans"])

        keypoints_by_id = {}
        #keypoints_by_id_frames = {}

        #print(indices)

        for frame_nr, keypoints_dict in enumerate(keypoints_along_frames):
            person_ids = list(keypoints_dict.keys())

            for p_id in person_ids:
                #print(person_ids)
                #print( keypoints_by_id)
                if p_id not in keypoints_by_id:
                    keypoints_by_id[p_id] = []
                    #keypoints_by_id_frames[p_id] = []
                
                keypoints_by_id[p_id].append((keypoints_dict[p_id], frame_nr)) # TODO: what if detection fails in one or several frames?
                #keypoints_by_id_frames[p_id].append(frame_nr)

        # heuristics:
        # A: Handshake / HandClap, 
        # B: Human touches the object ()

        # C: Humans look at each other
        # D: Pointing of Human
        # [x] E: Human starts walking 
        # [x] F: Human stops walking
        # G: Human Inside Car: complete skeleton inside bounding box
        # H: 
        # I: 
        # J: 

        # K
        # L
        # M

        print("Analyzed Human Skeleton IDs: ", keypoints_by_id.keys())
        ######################### Velocity ############################
        
        fig = plt.figure(figsize= [10, 15])
        
        joints_for_plotting = ['Spine', 'Pelvis', 'Thorax']
        joint_tmp = ['Spine', 'Pelvis', 'Thorax'][1] # 'Pelvis'# 'Thorax'# "Pelvis"# 'Spine'# 'Pelvis'
        wl = 60

        down_sampling = 60

        interval_length = 1 / fps * down_sampling

        event_log = []
        event_log_dict = {
            "initial_human_pose": [],
            "initial_object_pose": [],
            "person_starts_moving": [],
            "person_stops_moving": [],
            "hand_clap": []
        }

        for person_id in ['1', '2']: # keypoint_ids:
            ax = fig.add_subplot(len(keypoint_ids),1, int(person_id))
             # person_id = '1'
            keypoints_tuple = keypoints_by_id[person_id]

            keypoints_tmp = [k[0] for k in keypoints_tuple]

            keypoints_frames = [k[1] for k in keypoints_tuple]

            print(len(keypoints_tmp))

            # for k in test_keypoints[:5]:
            #     print("Pelvis: ", np.array(k[14])/1000)

            pelvic_pos_3d = []

            print("####################################### Keypoint Shape: ", np.array(keypoints_tmp).shape)

            for pos in keypoints_tmp:
                pelvic_pos_3d.append(pos[joints_name_dict[joint_tmp]]) # Pelvis

            print("Single Joint Array: ", np.array(pelvic_pos_3d).shape)

            fps = 60 / (int(indices[1]) - int(indices[0]))
            print("FPS: ", fps)

            print(pelvic_pos_3d[:5])

            time_delta = 1/fps * down_sampling

            pelvic_pos_3d = np.array(pelvic_pos_3d) / 1000
            #signals = [moving_average(pelvic_pos_3d[:, 0], wl), moving_average(pelvic_pos_3d[:, 1], wl), moving_average(pelvic_pos_3d[:, 2], wl)]
            

            from scipy.signal import butter, lfilter, freqz

            def butter_lowpass(cutoff, fs, order=5):
                return butter(order, cutoff, fs=fs, btype='low', analog=False)

            def butter_lowpass_filter(data, cutoff, fs, order=5):
                b, a = butter_lowpass(cutoff, fs, order=order)
                y = lfilter(b, a, data)
                return y

            # Filter requirements.
            order = 6
            fs = fps      # sample rate, Hz
            cutoff = 20 #3.667  # desired cutoff frequency of the filter, Hz

            # Get the filter coefficients so we can check its frequency response.
            b, a = butter_lowpass(cutoff, fs, order)

            # y = butter_lowpass_filter(data, cutoff, fs, order)

            # https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
            x = pelvic_pos_3d[:, 0]
            y = pelvic_pos_3d[:, 1]
            z = pelvic_pos_3d[:, 2]

            pelvic_pos_filtered = np.zeros_like(pelvic_pos_3d)
            pelvic_pos_filtered[:, 0] = butter_lowpass_filter(x, cutoff, fs, order)
            pelvic_pos_filtered[:, 1] = butter_lowpass_filter(y, cutoff, fs, order)
            pelvic_pos_filtered[:, 2] = butter_lowpass_filter(z, cutoff, fs, order)

            # median
            # signals = [scp_mf(pelvic_pos_3d[:, 0], wl), scp_mf(pelvic_pos_3d[:, 1], wl), scp_mf(pelvic_pos_3d[:, 2], wl)]

            def moving_average(x, w):
                average = np.convolve(x, np.ones(w), 'same') / w
                average[:w] = average[w]
                average[-w:] = average[-w]
                return average

            pelvic_pos_average = np.zeros_like(pelvic_pos_3d)
            pelvic_pos_average[:, 0] = moving_average(x, wl)
            pelvic_pos_average[:, 1] = moving_average(y, wl)
            pelvic_pos_average[:, 2] = moving_average(z, wl)

            print(pelvic_pos_filtered.shape)

            # downsample
            pelvic_pos_3d = pelvic_pos_3d[::down_sampling]
            pelvic_pos_filtered = pelvic_pos_filtered[::down_sampling]
            pelvic_pos_average = pelvic_pos_average[::down_sampling]

            pelvic_pos_diff = np.diff(pelvic_pos_3d, axis=0)
            pelvic_pos_diff_filtered = np.diff(pelvic_pos_filtered, axis=0) # / 1000

            pelvic_pos_diff_average = np.diff(pelvic_pos_average, axis=0)
            print("pelvic_pos_diff.shape: ", pelvic_pos_diff.shape)
            print(pelvic_pos_diff)
            print("Averaged Pose: ", pelvic_pos_diff_average)

            pelvic_velo_filtered = np.linalg.norm(pelvic_pos_diff_filtered, axis=1) / time_delta
            pelvic_velo_average = np.linalg.norm(pelvic_pos_diff_average, axis=1) / time_delta
            pelvic_velo = np.linalg.norm(pelvic_pos_diff, axis=1) / time_delta

            print("Velocity Shape: ", pelvic_velo.shape)
            #pelvic_velo = np.concatenate([np.array([pelvic_velo[0],]), pelvic_velo])
            # pelvic_velo_filtered = np.concatenate([np.array([pelvic_velo_filtered[0],]), pelvic_velo_filtered])
            # pelvic_velo_average = np.concatenate([np.array([pelvic_velo_average[0],]), pelvic_velo_average])

            pelvic_velo_grad = np.gradient(pelvic_pos_average, time_delta, axis=1)
            print("Grad", pelvic_velo_grad.shape)


            threshold_velo = 0.25
            moving_standing = np.zeros_like(pelvic_velo_average)

            moving_standing[pelvic_velo_average>threshold_velo] = 1

            #pelvic_velo = moving_average(pelvic_velo, int(fps))

            # pelvic_velo
            #pelvic_pos_3d
            
            print(pelvic_pos_3d.shape)
            # wl = 20
            #signals = [moving_average(pelvic_pos_3d[:, 0], 3), moving_average(pelvic_pos_3d[:, 1], 3), moving_average(pelvic_pos_3d[:, 2], 3)]
            #signals = [moving_average(pelvic_pos_3d[:, 0], wl), moving_average(pelvic_pos_3d[:, 1], wl), moving_average(pelvic_pos_3d[:, 2], wl)]
            #signals = [pelvic_pos_3d[:, 0], pelvic_pos_3d[:, 1], pelvic_pos_3d[:, 2]]
            signals = [pelvic_pos_average[:, 0], pelvic_pos_average[:, 1], pelvic_pos_average[:, 2]]
            time_axis = np.array(keypoints_frames)[::down_sampling] / fps
            ax.plot(time_axis, pelvic_pos_3d[:, 0])
            ax.plot(time_axis, pelvic_pos_3d[:, 1])
            ax.plot(time_axis, pelvic_pos_3d[:, 2])

            ax.plot(time_axis, signals[0])
            ax.plot(time_axis, signals[1])
            ax.plot(time_axis, signals[2])

            ax.plot(time_axis[1:], pelvic_velo)
            ax.plot(time_axis[1:], pelvic_velo_average)

            #ax.plot(time_axis[1:], pelvic_pos_diff)
            #ax.plot(time_axis[::down_sampling], pelvic_velo_average)
            #ax.plot(time_axis, pelvic_velo_grad)
            # ax.plot(time_axis[::down_sampling], pelvic_velo_filtered)
            # ax.plot(time_axis, moving_average(pelvic_velo, wl)[wl:-wl])

            ax2 = ax.twinx()
            ax2.set_ylabel('Moving/Standing')
            ax2.plot(time_axis[1:], moving_standing, 'k')


            event_type = 'initial_human_pose'
            event_data = (person_id, keypoints_tmp[0])
            time_stamp = time_axis[0]
            event_log.append(Event(event_type, event_data, time_stamp, keypoints_frames[0]))

            for i in range(1,len(moving_standing)):
                if moving_standing[i] == 1:
                    plt.axvspan(time_axis[i+1]-0.5*interval_length, time_axis[i+1]+0.5*interval_length, color='g', alpha=0.2, lw=0)

                # person starts moving
                if moving_standing[i-1] == 0 and moving_standing[i] == 1:
                    
                    event_type = 'person_starts_moving'
                    event_data = (person_id, keypoints_tmp[(i-1) * down_sampling])
                    time_stamp = time_axis[i-1]
                    event_log.append(Event(event_type, event_data, time_stamp, (i-1) * down_sampling))
                
                # person stops moving
                if moving_standing[i-1] == 1 and moving_standing[i] == 0:

                    event_type = 'person_stops_moving'
                    event_data = (person_id, keypoints_tmp[i * down_sampling])
                    time_stamp = time_axis[i]
                    event_log.append(Event(event_type, event_data, time_stamp, i * down_sampling))

            ax2.set_yticks([0,1])
            ax2.set_yticklabels(['Standing','Moving'])
        
            ax.set_xlabel("time [s]")
            ax.set_ylabel(f"[{joint_tmp}]joint velocity [m/s]")
            ax.title.set_text(f'{joint_tmp} Joint Velocity: Skeleton [{person_id}] | F:{wl}')
            ax.legend(['X', 'Y', 'Z','X_f', 'Y_f', 'Z_f', 'Velo', 'Velo_f'])

        # Hand Joint Detection

        # For Plotting
        date = datetime.now().strftime("%Y%m%d_%I%M%S")

        folder_name = input_file.split('-')[-2]

        output_path = f"./extracted_keyframes/{date}-" + folder_name + "-key_frames" #

        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

        plt.savefig(output_path + f"/{joint_tmp}_joint_velocity_subsampled.pdf")

        
        ############################## Hand Joint Distances ################################

        joints_name_dict['L_Hand']
        joints_name_dict['R_Hand']

        human_id_combinations = itertools.combinations(keypoint_ids, 2)

        print(list(human_id_combinations))

        for c in human_id_combinations:
            pass

        hand_distances = []

        for frame_nr, keypoints_dict in enumerate(keypoints_along_frames):
            right_hand_joint_1 = keypoints_dict['1'][joints_name_dict['R_Hand']]
            right_hand_joint_2 = keypoints_dict['2'][joints_name_dict['R_Hand']]
            # print("RHJ1: ", right_hand_joint_1)
            # print("RHJ2: ", right_hand_joint_2)

            h_distance = np.linalg.norm(np.array(right_hand_joint_1) - np.array(right_hand_joint_2))

            hand_distances.append(h_distance)

        fig = plt.figure(figsize= [10, 5])

        ax = fig.add_subplot(1,1,1)

        ax.set_xlabel("time [s]")
        joint_tmp = 'R_Hand'
        ax.set_ylabel(f"[{joint_tmp}] joint distance [mm]")
        ax.title.set_text(f'{joint_tmp} Joint Distance: Skeleton [1/2]')

        time_axis = np.arange(len(keypoints_along_frames))/fps

        hand_distances = np.array(hand_distances)
        hand_distances_smoothed = moving_average(hand_distances, 3)
        plt.plot(time_axis, hand_distances)
        plt.plot(time_axis, hand_distances_smoothed)

        handshake_threshold = 250
        interval_length = 1 / fps

        only_one = False
        for i in range(1,len(hand_distances)):
            if hand_distances_smoothed[i] < handshake_threshold:
                plt.axvspan(time_axis[i]-0.5*interval_length, time_axis[i]+0.5*interval_length, color='g', alpha=0.2, lw=0)


                if only_one is False:  # TODO: check for first event of longer period
                    event_type = 'hand_clap'
                    event_data = {
                        '1': list(keypoints_along_frames[i]['1']),
                        '2': list(keypoints_along_frames[i]['2'])
                    }
                    print("Event_data: ", event_data)
                    time_stamp = time_axis[i]
                    event_log.append(Event(event_type, event_data, time_stamp, i))

                    only_one = True
        print(type(keypoints_along_frames[i]['1']))

        print(min(hand_distances))

        plt.savefig(output_path + "/hand_joint_distance.pdf")


        # save events as json


        ############################## Human Object Distances ################################
        detect_human_objct_relation = False

        cars = [Detection9D(**obj) for obj in pred_objects if obj['class_name'] == 'car']

        for idx, car in enumerate(cars):
            event_type = 'initial_object_pose'
            
            tensor = car.get_tensor()
            tensor[2] = tensor[2] / 1.87
            # tensor = tensor[]
            boxes = CameraInstance3DBoxes([np.take(tensor, [0,1,2,3,4,5,7])], box_dim=7)
            boxes_corners = boxes.corners
            corners = boxes_corners[0].tolist()
            event_data = (idx, tensor) # corners
            time_stamp = idx 
            event_log.append(Event(event_type, event_data, time_stamp, bbox_frame_index))

        print(event_log)

        # save event log
        log_file_name = output_path + '/' + folder_name + '-event_log.json'
        with open(log_file_name, "wt") as file:
            json.dump(event_log, file, default=_to_dict) 

        if detect_human_objct_relation:
            # cars = [Detection9D(**obj) for obj in pred_objects if obj['class_name'] == 'car']

            print(cars)

            def closestPointToBox(q, origin, v100, v010, v001):
                # (Vector3 q, Vector3 origin, Vector3 v100, Vector3 v010, Vector3 v001)

                px = v100
                py = v010
                pz = v001

                vx = (px - origin)
                vy = (py - origin)
                vz = (pz - origin)

                tx = np.dot( q - origin, vx ) / np.dot(vx,vx)
                ty = np.dot( q - origin, vy ) / np.dot(vy,vy)
                tz = np.dot( q - origin, vz ) / np.dot(vz,vz)

                # tx = tx < 0 ? 0 : tx > 1 ? 1 : tx;
                if tx < 0:
                    tx = 0
                elif tx > 1:
                    tx = 1
                # else:
                    # tx = tx
                
                # ty = ty < 0 ? 0 : ty > 1 ? 1 : ty;
                if ty < 0:
                    ty = 0
                elif ty > 1:
                    ty = 1
                
                # tz = tz < 0 ? 0 : tz > 1 ? 1 : tz;
                if tz < 0:
                    tz = 0
                elif tz > 1:
                    tz = 1

                p = tx * vx + ty * vy + tz * vz + origin

                return p

            first_car = cars[0]

            tensor = first_car.get_tensor()
            tensor[2] = tensor[2] / 1.87
            # tensor = tensor[]
            boxes = CameraInstance3DBoxes([np.take(tensor, [0,1,2,3,4,5,7])], box_dim=7)
            boxes_corners = boxes.corners

            print(boxes_corners)

                #  (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
                # 
                #              front z
                #                   /
                #                  /
                #    (x0, y0, z1) + -----------  + (x1, y0, z1)
                #                /|            / |
                #               / |           /  |
                # (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                #              |  /      .   |  /
                #              | / origin    | /
                # (x0, y1, z0) + ----------- + -------> x right
                #              |             (x1, y1, z0)
                #              |
                #              v
                #         down y
            
            q = np.array(keypoints_dict['1'][joints_name_dict['Pelvis']]) / 1000
            origin = boxes_corners[0][3].numpy()
            v100 = boxes_corners[0][7].numpy()
            v010 = boxes_corners[0][0].numpy()
            v001 = boxes_corners[0][2].numpy()

            p = closestPointToBox(q, origin, v100, v010, v001)

            print(p)

            distance_human_object = np.linalg.norm(q - p)

            print(distance_human_object)

            def get_distance_human_object(human_id = '1', joint_name = 'Pelvis', obj=pred_objects[0]):
                
                distances = []
                for frame_nr, keypoints_dict in enumerate(keypoints_along_frames):
                    queried_joint = np.array(keypoints_dict['1'][joints_name_dict[joint_name]])


                    tensor = obj.get_tensor()
                    tensor[2] = tensor[2] / 1.87
                    # tensor = tensor[]
                    boxes = CameraInstance3DBoxes([np.take(tensor, [0,1,2,3,4,5,7])], box_dim=7)
                    boxes_corners = boxes.corners

                    q = queried_joint / 1000
                    origin = boxes_corners[0][3].numpy()
                    v100 = boxes_corners[0][7].numpy()
                    v010 = boxes_corners[0][0].numpy()
                    v001 = boxes_corners[0][2].numpy()

                    p = closestPointToBox(q, origin, v100, v010, v001)
                    distance_human_object = np.linalg.norm(q - p)

                    distances.append(distance_human_object)

                return distances
                    # h_distance = np.linalg.norm(np.array(right_hand_joint_1) - np.array(right_hand_joint_2))

                    #hand_distances.append(h_distance)

            car_distances = get_distance_human_object(human_id = '1', joint_name = 'Pelvis', obj=cars[1]) # TODO: loop efficiently through all objects and humans

            fig = plt.figure(figsize= [10, 5])

            ax = fig.add_subplot(1,1,1)

            ax.set_xlabel("time [s]")
            joint_tmp = 'Pelvis'
            ax.set_ylabel(f"[{joint_tmp}] joint distance to car [mm]")
            #ax.title.set_text(f'{joint_tmp} Joint Distance: Skeleton [1/2]')

            time_axis = np.arange(len(keypoints_along_frames))/fps

            car_distances = np.array(car_distances)# .T
            print(car_distances)
            car_distances_smoothed = moving_average(car_distances, 3)
            plt.plot(time_axis, car_distances)
            plt.plot(time_axis, car_distances_smoothed)

            # handshake_threshold = 250
            # interval_length = 1 / fps
            # for i in range(1,len(hand_distances)):
            #     if hand_distances_smoothed[i] < handshake_threshold:
            #         plt.axvspan(time_axis[i]-0.5*interval_length, time_axis[i]+0.5*interval_length, color='g', alpha=0.2, lw=0)

            print(min(hand_distances))

            plt.savefig(output_path + "/human_object_distance.pdf")