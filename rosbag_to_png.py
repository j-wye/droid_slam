import os
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # 변경된 부분

class ImageExtractor(Node):
    def __init__(self):
        super().__init__('image_extractor')
        self.subscription = self.create_subscription(
            Image,  # 변경된 부분
            '/oakd/rgb/preview/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.image_counter = 0

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # 변경된 부분
        image_path = f'turtle/{self.image_counter:06d}.png'
        cv2.imwrite(image_path, cv_image)
        self.image_counter += 1
        self.get_logger().info(f'Saved {image_path}')

def main(args=None):
    rclpy.init(args=args)
    image_extractor = ImageExtractor()
    rclpy.spin(image_extractor)
    image_extractor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    os.makedirs('turtle', exist_ok=True)
    main()