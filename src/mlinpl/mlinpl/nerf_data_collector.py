#!/usr/bin/env python3

import os
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import TransformListener, Buffer
from cv_bridge import CvBridge


class NerfDataCollector(Node):
    """
    Simplified ROS2 Node that collects images and camera poses for NeRF training.
    """
    
    def __init__(self):
        super().__init__('nerf_data_collector')
        
        # Declare parameters
        self.declare_parameter('image_topic', '/rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/rgb/camera_info')
        self.declare_parameter('source_frame', 'base_link')
        self.declare_parameter('target_frame', 'azure_rgb')
        self.declare_parameter('output_dir', 'nerf_data')
        self.declare_parameter('collection_rate', 10)
        
        # Get parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.source_frame = self.get_parameter('source_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.output_dir = self.get_parameter('output_dir').value
        self.collection_rate = self.get_parameter('collection_rate').value
        
        # Initialize
        self.bridge = CvBridge()
        self.camera_params = None
        self.frames_data = []
        self.frame_count = 0
        self.i = 0
        self.collecting = False
        self.start_timer_called = False
        self.latest_image = None
        
        # Create output directory
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 1
        )
        
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 1
        )
        
        # Collection timer
        # self.collection_timer = self.create_timer(1.0 / self.collection_rate, self.collect_frame)
        
        self.get_logger().info(f"NeRF Data Collector initialized")
        self.get_logger().info(f"Will collect at {self.collection_rate} Hz")
    
    def image_callback(self, msg):
        """Store the latest image."""
        if not self.collecting:
            self.get_logger().warn("Not collecting yet, waiting for camera info")
            return

        if self.i % self.collection_rate == self.collection_rate - 1:  # Collect every nth image
            self.latest_image = msg
            self.collect_frame()

        self.i += 1
    
    def camera_info_callback(self, msg):
        """Get camera parameters once."""
        if self.camera_params is None:
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d)
            
            self.camera_params = {
                "camera_model": "OPENCV",
                "fl_x": K[0, 0],
                "fl_y": K[1, 1], 
                "cx": K[0, 2],
                "cy": K[1, 2],
                "w": msg.width,
                "h": msg.height,
                # "k1": D[0] if len(D) > 0 else 0.0,
                # "k2": D[1] if len(D) > 1 else 0.0,
                # "p1": D[2] if len(D) > 2 else 0.0,
                # "p2": D[3] if len(D) > 3 else 0.0
            }
            self.get_logger().info("Camera parameters received")
            self.collecting = True
            self.get_logger().info("Starting collection")
    
    def collect_frame(self):
        """Collect one frame (image + pose)."""
        
        if self.latest_image is None:
            self.get_logger().warn("No image received yet")
            return
        
        try:
            # Use the latest image
            image_msg = self.latest_image
            
            # Get transform for image timestamp
            transform = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.target_frame,
                image_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Save image
            cv_img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            # cv_img = cv2.resize(cv_img, (640, 480))
            image_filename = f"frame_{self.frame_count:06d}.jpg"
            image_path = os.path.join(self.images_dir, image_filename)
            cv2.imwrite(image_path, cv_img)
            
            # Convert transform to NeRF format
            nerf_transform = self.ros_to_nerf_transform(transform)
            
            # Store frame data
            frame_data = {
                "file_path": f"images/{image_filename}",
                "transform_matrix": nerf_transform.tolist()
            }
            self.frames_data.append(frame_data)
            
            self.frame_count += 1
            self.get_logger().info(f"Collected frame {self.frame_count}")
            
        except Exception as e:
            self.get_logger().warn(f"Failed to collect frame: {str(e)}")
    
    def ros_to_nerf_transform(self, tf_transform):
        """Convert ROS transform to NeRF Studio format."""
        # Extract pose from ROS transform
        t = tf_transform.transform
        translation = np.array([t.translation.x, t.translation.y, t.translation.z])
        rotation = Rotation.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
        
        # Create 4x4 homogeneous matrix (camera-to-world)
        T_ros = np.eye(4)
        rot_matrix = rotation.as_matrix()
        # Normalize the rotation matrix to ensure orthogonality
        u, _, vh = np.linalg.svd(rot_matrix)
        rot_matrix_normalized = np.dot(u, vh)
        T_ros[:3, :3] = rot_matrix_normalized
        T_ros[:3, 3] = translation

        ros_to_nerf = np.array([
            [1, 0,  0, 0],
            [0,  -1,  0, 0],
            [0, 0,  -1, 0],
            [0,  0,  0, 1]
        ])

        # Apply coordinate transformation
        T_nerf = T_ros @ ros_to_nerf
        
        return T_nerf  # Return full 4x4 matrix
    
    def finish_collection(self):
        """Save transforms.json and exit."""
        self.collecting = False
        
        if self.frame_count == 0:
            self.get_logger().error("No frames collected!")
            return
        
        # Create NeRF Studio data structure
        nerf_data = {
            **self.camera_params,
            "frames": self.frames_data
        }
        
        # Save transforms.json
        transforms_path = os.path.join(self.output_dir, "transforms.json")
        with open(transforms_path, 'w') as f:
            json.dump(nerf_data, f, indent=2)
        
        self.get_logger().info(f"Collection complete! Saved {self.frame_count} frames to {self.output_dir}")
        
        # Shutdown
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        collector = NerfDataCollector()
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.finish_collection()
    finally:
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
