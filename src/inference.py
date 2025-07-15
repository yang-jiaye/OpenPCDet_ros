#! /usr/bin/env python3.8
"""
Created on Thu Aug  6 11:27:43 2020

@author: Javier del Egido Sierra and Carlos Gómez-Huélamo

===

Modified on 23 Dec 2022
@author: Kin ZHANG (https://kin-zhang.github.io/)

Part of codes also refers: https://github.com/kwea123/ROS_notes
"""

# General use imports
import os
import time
import glob
from pathlib import Path
import gc

# ROS imports
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from v2x_msgs.msg import RoadCarMsg

# Math and geometry imports
import math
import numpy as np
import torch

# CUDA memory management settings
torch.cuda.empty_cache()
if torch.cuda.is_available():
    # Set memory fraction to avoid fragmentation
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    # Enable memory growth
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# OpenPCDet imports
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

# Kin's utils
from utils.draw_3d import Draw3DBox
from utils.global_def import *
from utils import *

import yaml
import os

def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        memory_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - memory_allocated
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Free: {memory_free:.2f}GB")

def load_parameters():
    """Load parameters from ROS parameter server with fallback to YAML file"""
    BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
    
    # Load default parameters from YAML file
    with open(f"{BASE_DIR}/launch/config.yaml", 'r') as f:
        try:
            para_cfg = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            para_cfg = yaml.safe_load(f)
    
    # Override with ROS parameters if available
    node_name = rospy.get_name()
    
    cfg_root = rospy.get_param(f'{node_name}/cfg_root', para_cfg["cfg_root"])
    model_path = rospy.get_param(f'{node_name}/model_path', para_cfg["model_path"])
    move_lidar_center_x = rospy.get_param(f'{node_name}/move_lidar_center_x', para_cfg["move_lidar_center_x"])
    move_lidar_center_y = rospy.get_param(f'{node_name}/move_lidar_center_y', para_cfg["move_lidar_center_y"])
    height_addtion = rospy.get_param(f'{node_name}/height_addtion', para_cfg.get("height_addtion", 0.0))
    rx = rospy.get_param(f'{node_name}/rx', para_cfg["rx"])
    ry = rospy.get_param(f'{node_name}/ry', para_cfg["ry"])
    rz = rospy.get_param(f'{node_name}/rz', para_cfg["rz"])
    threshold = rospy.get_param(f'{node_name}/threshold', para_cfg["threshold"])
    pointcloud_topic = rospy.get_param(f'{node_name}/pointcloud_topic', para_cfg["pointcloud_topic"])
    viz_rate = rospy.get_param(f'{node_name}/viz_rate', para_cfg["viz_rate"])
    result_path = rospy.get_param(f'{node_name}/result_path', para_cfg["result_path"])

    logitude = rospy.get_param(f'{node_name}/logitude', para_cfg.get("logitude", 110.0))
    latitude = rospy.get_param(f'{node_name}/latitude', para_cfg.get("latitude", 20.0))
    height_global = rospy.get_param(f'{node_name}/height_global', para_cfg.get("height_global", 50.0))
    yaw = rospy.get_param(f'{node_name}/yaw', para_cfg.get("yaw", 0.0))
    
    return {
        'cfg_root': cfg_root,
        'model_path': model_path,
        'move_lidar_center_x': move_lidar_center_x,
        'move_lidar_center_y': move_lidar_center_y,
        'height_addtion': height_addtion,
        'rx': rx, 'ry': ry, 'rz': rz,
        'threshold': threshold,
        'pointcloud_topic': pointcloud_topic,
        'viz_rate': viz_rate,
        'result_path': result_path,
        'logitude': logitude,
        'latitude': latitude,
        'height_global': height_global,
        'yaw': yaw
    }

# Initialize ROS node first
rospy.init_node('object_3d_detector_node', anonymous=True)

# Load parameters
params = load_parameters()
cfg_root = params['cfg_root']
model_path = params['model_path']
move_lidar_center_x = params['move_lidar_center_x']
move_lidar_center_y = params['move_lidar_center_y']
height_addtion = params['height_addtion']
rx = params['rx']
ry = params['ry']
rz = params['rz']
threshold = params['threshold']
pointcloud_topic = params['pointcloud_topic']
RATE_VIZ = params['viz_rate']
result_path = params['result_path']
longitude = params['logitude']
latitude = params['latitude']
height_global = params['height_global']
yaw = params['yaw']
inference_time_list = []

# not used
def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id="livox_frame"):
    """
    Create a sensor_msgs.PointCloud2 from an array of points.
    """
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def transform_points(points):
        """
        Common point cloud transformation method
        Apply rotation and translation to input points
        """
        num_features = 4 # X,Y,Z,intensity       
        transformed_points = points.reshape([-1, num_features]).copy()

        def roty(t):
            """ Rotation about the y-axis. """
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        Ry = roty(np.deg2rad(ry))

        def rotx(t):
            """ Rotation about the x-axis. """
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        Rx = rotx(np.deg2rad(rx))

        def rotyz(t):
            """ Rotation about the z-axis. """
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Rz = rotyz(np.deg2rad(rz))

        # Apply rotation transformation: R * points.T -> (N, 3)
        # Combined rotation matrix: Rz * Rx * Ry
        R_combined = Rz @ Rx @ Ry
        transformed_points[:,0:3] = (R_combined @ transformed_points[:,0:3].T).T

        transformed_points[:,0] += move_lidar_center_x
        transformed_points[:,1] += move_lidar_center_y
        transformed_points[:,2] += height_addtion
        transformed_points[:,3] = transformed_points[:,3] / 255.0

        return transformed_points

def lan_lon_add_x_y(lat, lon, x, y, yaw):
    """
    根据经纬度和偏移量计算新的经纬度
    lat: 纬度
    lon: 经度
    x: 雷达系x
    y: 雷达系y
    yaw: 雷达系方向角, x轴 北偏东
    返回新的经纬度 (latitude, longitude)
    """
    R = 6378137.0  # 地球半径，单位为米
    dN = x * np.cos(np.radians(yaw)) - y * np.sin(np.radians(yaw))  # 东向偏移
    dE = x * np.sin(np.radians(yaw)) + y * np.cos(np.radians(yaw))  # 北向偏移 

    d_lat = dN / R  # 纬度变化量
    d_lon = dE / (R * np.cos(np.pi * lat / 180))  # 经度变化量

    new_lat = lat + d_lat * (180 / np.pi)
    new_lon = lon + d_lon * (180 / np.pi)

    return new_lat, new_lon

def rslidar_callback(msg):
    global threshold, move_lidar_center_x, move_lidar_center_y, height_addtion, rx, ry, rz, RATE_VIZ, proc_1
    
    try:
        # Clear CUDA cache at the beginning of each callback
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update parameters dynamically from ROS parameter server
        node_name = rospy.get_name()
        threshold = rospy.get_param(f'{node_name}/threshold', threshold)
        move_lidar_center_x = rospy.get_param(f'{node_name}/move_lidar_center_x', move_lidar_center_x)
        move_lidar_center_y = rospy.get_param(f'{node_name}/move_lidar_center_y', move_lidar_center_y)
        height_addtion = rospy.get_param(f'{node_name}/height_addtion', height_addtion)
        rx = rospy.get_param(f'{node_name}/rx', rx)
        ry = rospy.get_param(f'{node_name}/ry', ry)
        rz = rospy.get_param(f'{node_name}/rz', rz)
        y_min = rospy.get_param(f'{node_name}/y_min', -10)
        y_max = rospy.get_param(f'{node_name}/y_max', 5)

        RATE_VIZ = rospy.get_param(f'{node_name}/viz_rate', RATE_VIZ)
        
        select_boxs, select_types = [],[]
        if proc_1.no_frame_id:
            proc_1.set_viz_frame_id(msg.header.frame_id)
            print(f"{bc.OKGREEN} setting marker frame id to lidar: {msg.header.frame_id} {bc.ENDC}")
            proc_1.no_frame_id = False

        frame = msg.header.seq # frame id -> not timestamp
        print(frame)
        msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        np_p = get_xyz_points(msg_cloud, True)

        transformed_points = transform_points(np_p)
        
        # # Fast point cloud transformation and publishing (high frequency)
        proc_1.publish_transformed_points(transformed_points, frame)
        
        # Publish coordinate axes for visualization
        proc_1.publish_3d_axes(msg.header.frame_id)
        
        # inference here (potentially slower, won't affect point cloud publishing)
        scores, dt_box_lidar, types, pred_dict = proc_1.run(transformed_points, frame, timestamp = msg.header.stamp.to_sec())

        # select confident predictions
        for i, score in enumerate(scores):
            if score>threshold and dt_box_lidar[i][1] < y_max and dt_box_lidar[i][1] > y_min:
                if pred_dict['name'][i] == 'Car':
                # if score>threshold:
                    select_boxs.append(dt_box_lidar[i])
                    select_types.append(pred_dict['name'][i])


        # publish 3d box
        if(len(select_boxs)>0):
            proc_1.pub_rviz.publish_3dbox(np.array(select_boxs), -1, pred_dict['name'])
            print_str = f"Frame id: {frame}. Prediction results: \n"
            for i in range(len(pred_dict['name'])):
                print_str += f"Type: {pred_dict['name'][i]:.3s} Prob: {scores[i]:.2f}, Pos: {dt_box_lidar[i][0]:.2f} {dt_box_lidar[i][1]:.2f} {dt_box_lidar[i][2]:.2f} \n" 
            print(print_str)
            # with open(result_path, "a") as file:
            #     for i in range(len(select_types)):
            #         dt_box_str = ' '.join(str(x) for x in dt_box_lidar[i])
            #         file.write(f"{frame} {select_types[i]} " + dt_box_str + '\n')

            # Only publish if there are predictions and timestamp is provided
            proc_1.publish_v2xmsg(select_boxs, timestamp=msg.header.stamp.to_sec())

        else:
            print(f"\n{bc.FAIL} No confident prediction in this time stamp {bc.ENDC}\n")
        print(f" -------------------------------------------------------------- ")
        
        # Clean up variables to free memory
        del transformed_points, np_p, msg_cloud
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"{bc.FAIL}CUDA out of memory error: {e}{bc.ENDC}")
            print(f"{bc.WARNING}Trying to clear cache and continue...{bc.ENDC}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            return
        else:
            raise e

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        self.pub_rviz = None
        self.no_frame_id = True
        self.rate = RATE_VIZ
        self.pub_pts = rospy.Publisher("rotated_points", PointCloud2, queue_size=10)
        self.v2x_pub = rospy.Publisher('/road_car_topic', RoadCarMsg, queue_size=10)
        self.pub_axes = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    def publish_3d_axes(self, frame_id='velodyne'):
        """
        Publish 3D axes for visualization in RViz
        """
        from geometry_msgs.msg import Point
        
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "axes"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        node_name = rospy.get_name()
        x_m = rospy.get_param(f'{node_name}/x_m', 20)
        y_m = rospy.get_param(f'{node_name}/y_m', -5)
        z_m = rospy.get_param(f'{node_name}/z_m', -1)
        
        # X axis (red) - arrow from origin to (x_m+5, y_m, z_m)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.2  # Arrow shaft diameter
        marker.scale.y = 0.4  # Arrow head diameter
        marker.scale.z = 1.0  # Arrow head length
        
        # Define arrow points: start point (origin) and end point
        start_point = Point()
        start_point.x = x_m
        start_point.y = y_m
        start_point.z = z_m
        
        end_point = Point()
        end_point.x = x_m + 5.0  # 5 meter arrow in X direction
        end_point.y = y_m
        end_point.z = z_m
        
        marker.points = [start_point, end_point]
        
        # Publish the marker
        self.pub_axes.publish(marker)

    def set_pub_rviz(self, box3d_pub, marker_frame_id = 'velodyne'):
        self.pub_rviz = Draw3DBox(box3d_pub, marker_frame_id, self.rate)
    
    def set_viz_frame_id(self, marker_frame_id):
        self.pub_rviz.set_frame_id(marker_frame_id)
 
    def publish_transformed_points(self, points, frame):
        """
        Fast transformation and publishing of point cloud without inference
        This ensures high-frequency point cloud publishing
        """
        # Use shared transformation method
        transformed_points = points

        # Publish immediately without waiting for inference
        msg = xyz_array_to_pointcloud2(transformed_points)
        self.pub_pts.publish(msg)

    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path("/home/ackerman/Workspace/Archive/OpenPCDet/tools/000008.bin"),
            ext='.bin')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        print("Model path: ", self.model_path)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

    def get_template_prediction(self, num_samples):
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict
    
    def publish_v2xmsg(self, select_boxes, timestamp):
        """
        Publish V2X message with prediction results
        """

        box = select_boxes[0]
        # pos 为 box 中心
        pos = box[:3]
        print(f"车辆在雷达系下坐标: [{pos[0]}, {pos[1]}, {pos[2]}]， 距离: {math.sqrt(pos[0]**2 + pos[1]**2):.2f} m")
        # Create RoadCarMsg

        road_car_msg = RoadCarMsg()

        plate_no = "AAABBB"
        id_bytes = plate_no.encode('utf-8')[:8]  # 取前8个字节
        id_array = list(id_bytes) + [0] * (8 - len(id_bytes))  # 如果不足8字节则用0填充
        road_car_msg.ID = id_array

        node_name = rospy.get_name()
        # Use global variables as defaults, but assign to local variables with different names
        current_longitude = rospy.get_param(f'{node_name}/logitude', longitude)
        current_latitude = rospy.get_param(f'{node_name}/latitude', latitude)
        current_height_global = rospy.get_param(f'{node_name}/height_global', height_global)
        current_yaw = rospy.get_param(f'{node_name}/yaw', yaw)

        lat, lon = lan_lon_add_x_y(
            current_latitude, current_longitude, pos[0], pos[1], current_yaw
        )

        road_car_msg.latitude = int(lat * 1e8)
        road_car_msg.longitude = int(lon * 1e8)
        road_car_msg.height = int((current_height_global + pos[2]) * 1e3)
        print(f"发布车牌号: {plate_no}, 经纬度: ({road_car_msg.latitude}, {road_car_msg.longitude}), 高度: {road_car_msg.height}")

        road_car_msg.secs = int(timestamp)
        road_car_msg.nsecs = int((timestamp - road_car_msg.secs) * 1e9)

        road_car_msg.accident_tp = 10

        print(f"发布时间戳: ：：：：：：：：{road_car_msg.secs}.{road_car_msg.nsecs}")

        self.v2x_pub.publish(road_car_msg)
        

    def run(self, points, frame, timestamp=None):
        try:
            t_t = time.time()
            
            # Use shared transformation method instead of duplicating code
            self.points = points

            input_dict = {
                'points': self.points,
                'frame_id': frame,
            }

            data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
            data_dict = self.demo_dataset.collate_batch([data_dict])
            
            # Move data to GPU with memory management
            with torch.cuda.device(self.device):
                load_data_to_gpu(data_dict)

                torch.cuda.synchronize()
                t = time.time()

                # inference here with torch.no_grad() to save memory
                with torch.no_grad():
                    pred_dicts, _ = self.net.forward(data_dict)
                
                torch.cuda.synchronize()
                inference_time = time.time() - t
                inference_time_list.append(inference_time)
                mean_inference_time = sum(inference_time_list)/len(inference_time_list)

                # Move results to CPU immediately to free GPU memory
                boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
                scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
                types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()
                
                # Clear GPU tensors immediately
                del pred_dicts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pred_boxes = np.copy(boxes_lidar)
            pred_dict = self.get_template_prediction(scores.shape[0])

            pred_dict['name'] = np.array(cfg.CLASS_NAMES)[types - 1]
            pred_dict['score'] = scores
            pred_dict['boxes_lidar'] = pred_boxes

            # if len(pred_boxes) > 0 and timestamp is not None:
            # # Only publish if there are predictions and timestamp is provided
            #     self.publish_v2xmsg(pred_dict, timestamp)

            return scores, boxes_lidar, types, pred_dict
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"{bc.FAIL}CUDA out of memory in inference: {e}{bc.ENDC}")
                # Emergency cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                # Return empty results
                return np.array([]), np.array([]), np.array([]), {'name': np.array([]), 'score': np.array([]), 'boxes_lidar': np.array([])}
            else:
                raise e
    
 
if __name__ == "__main__":
    print("Starting OpenPCDet ROS Node...")
    print_gpu_memory_usage()
    
    no_frame_id = False
    proc_1 = Processor_ROS(cfg_root, model_path)
    print(f"\n{bc.OKCYAN}Config path: {bc.BOLD}{cfg_root}{bc.ENDC}")
    print(f"{bc.OKCYAN}Model path: {bc.BOLD}{model_path}{bc.ENDC}")
    
    print("Initializing model...")
    proc_1.initialize()
    print("Model initialized successfully!")
    print_gpu_memory_usage()
    
    # Node already initialized at the beginning
    sub_lidar_topic = [pointcloud_topic]

    cfg_from_yaml_file(cfg_root, cfg)
    
    sub_ = rospy.Subscriber(sub_lidar_topic[0], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    pub_rviz = rospy.Publisher('detect_3dbox',MarkerArray, queue_size=10)
    proc_1.set_pub_rviz(pub_rviz)
    print(f"{bc.HEADER} ====================== {bc.ENDC}")
    print(" ===> [+] PCDet ros_node has started. Try to Run the rosbag file")
    print(f"{bc.HEADER} ====================== {bc.ENDC}")

    rospy.spin()
