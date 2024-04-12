import rclpy, sys, torch, lietorch, cv2, os, glob, time, argparse
sys.path.append('droid_slam')
sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
from tqdm import tqdm
import numpy as np
from torch.multiprocessing import Process
from droid import Droid
import torch.nn.functional as F
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class DROIDSLAM_ROS2_SERIAL(Node):
    def __init__(self):
        super().__init__('droid_slam_ros2_node')
        self.bridge = CvBridge()
        self.qos_policy = QoSProfile(reliability = ReliabilityPolicy.RELIABLE,history = HistoryPolicy.KEEP_LAST, depth=1)
        self.img_sub = self.create_subscription(CompressedImage, '/oakd/rgb/image_raw/compressed', self.img_cb, self.qos_policy)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_cb, self.qos_policy)
        self.current_lidar_ranges = []

        self.frame_id = 0
        self.droid = None
        self.args = self.parse_arguments()   

    def lidar_cb(self, data):
        self.current_lidar_ranges = data.ranges
        self.lidar_angle_min = data.angle_min
        self.lidar_angle_max = data.angle_max
        self.lidar_angle_increment = data.angle_increment
        self.lidar_range_min = data.range_min
        self.lidar_range_max = data.range_max
        self.laser_point = round((self.lidar_angle_max - self.lidar_angle_min) / self.lidar_angle_increment)
        print(self.laser_point)

    def img_cb(self, data):
        try:
            # (720, 1280) : img size가 default
            cv_img = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        self.process_image(cv_img)
    
    def show_image(self, img):
        img = img.permute(1, 2, 0).cpu().numpy()
        cv2.imshow('image', img)
        cv2.waitKey(1)

    def process_image(self, img):
        # HERE! : Can modify image size which I want
        mod_height = 480
        mod_width = 640
        if self.droid is None:
            calib = np.loadtxt(self.args.calib, delimiter=" ")
            fx, fy, cx, cy = calib[:4]
            self.intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
            self.droid = Droid(self.args)
        
        mod_img = cv2.resize(img, (mod_width, mod_height))
        frame = cv2.cvtColor(mod_img, cv2.COLOR_BGR2RGB)
        frame = torch.as_tensor(frame).permute(2, 0, 1).to(torch.float32)
        
        if self.frame_id % self.args.stride == 0:
            self.droid.track(self.frame_id, frame[None], intrinsics=self.intrinsics)
            if not self.args.disable_vis:
                self.show_image(frame)
        self.frame_id += 1

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
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
        args.stereo = False
        torch.multiprocessing.set_start_method('spawn', force=True)
        return args

def main(args=None):
    rclpy.init(args=args)
    droid_slam_ros2 = DROIDSLAM_ROS2_SERIAL()
    rclpy.spin(droid_slam_ros2)
    droid_slam_ros2.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
