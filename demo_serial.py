import rclpy
import sys
sys.path.append('droid_slam')
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import argparse
from torch.multiprocessing import Process
from droid import Droid
import torch.nn.functional as F



from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class DROIDSLAM_ROS2(Node):
    def __init__(self):
        super().__init__('droid_slam_ros2_node')
        self.bridge = CvBridge()
        self.qos_policy = QoSProfile(reliability = ReliabilityPolicy.RELIABLE,history = HistoryPolicy.KEEP_LAST, depth=1)
        self.img_sub = self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.img_cb, self.qos_policy)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_cb, self.qos_policy)
        self.current_lidar_ranges = []

        self.droid = None
        self.args = self.parse_arguments()

        torch.multiprocessing.set_start_method('spawn', force=True)

    def lidar_cb(self, data):
        self.current_lidar_ranges = data.ranges

    def img_cb(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

    def process_image(self, img):
        if self.droid is None:
            self.droid = Droid(self.args)
        
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = torch.as_tensor(frame).permute(2, 0, 1).to(torch.float32)
        intrinsics = torch.tensor([self.args.fx, self.args.fy, self.args.cx, self.args.cy], dtype=torch.float32)

        # Assuming you have a method to keep track of frame id's
        frame_id = self.get_frame_id()
        self.droid.track(frame_id, frame[None], intrinsics=intrinsics)

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--calib", type=str, default="./calib/tartan.txt", help="path to calibration file")
        # Add other argument definitions here as needed, similar to the initial script.
        parser.add_argument("--calib", type=str, default="./calib/tartan.txt", help="path to calibration file")
        parser.add_argument("--stride", default=2, type=int, help="frame stride")

        parser.add_argument("--weights", default="droid.pth")
        parser.add_argument("--buffer", type=int, default=512)
        parser.add_argument("--image_size", default=[480, 640])
        parser.add_argument("--disable_vis", action="store_true")

        parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
        parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
        parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
        parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
        parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
        parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
        parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
        parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

        parser.add_argument("--backend_thresh", type=float, default=22.0)
        parser.add_argument("--backend_radius", type=int, default=2)
        parser.add_argument("--backend_nms", type=int, default=3)
        parser.add_argument("--upsample", action="store_true")
        parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
        
        args = parser.parse_args()
        return args

    def get_frame_id(self):
        # Implement a method to generate or retrieve the current frame id
        pass

def main(args=None):
    rclpy.init(args=args)
    droid_slam_ros = DROIDSLAM_ROS2()
    rclpy.spin(droid_slam_ros)
    droid_slam_ros.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()