#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import tf_transformations
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge


class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        
        self.bridge = CvBridge()
        
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10
        )
        
        self.marker_pub = self.create_publisher(MarkerArray, '/slam/aruco_markers', 10)
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        self.get_logger().info('SLAM Node initialized, waiting for camera info and data...')
    
    def camera_info_callback(self, msg):
        """Получение параметров камеры"""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info('Camera info received')
    
    def image_callback(self, msg):
        """Callback для обработки изображений"""
        if not self.camera_info_received:
            return
            
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            self.detect_aruco_markers(gray_image, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {str(e)}')
    
    def detect_aruco_markers(self, gray_image, header):
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray_image)

        if ids is not None and len(ids) > 0:
            marker_size = 0.1
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size, self.camera_matrix, self.dist_coeffs
                )

                markers_list = []
                marker_array = MarkerArray()

                for i in range(len(ids)):
                    marker_id = int(ids[i][0])
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]

                    R, _ = cv2.Rodrigues(rvec)

                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec
                    quat = tf_transformations.quaternion_from_matrix(T)

                    cube_marker = Marker()
                    cube_marker.header = header
                    cube_marker.header.frame_id = 'camera_link'
                    cube_marker.ns = "aruco_cubes"
                    cube_marker.id = marker_id
                    cube_marker.type = Marker.CUBE
                    cube_marker.action = Marker.ADD

                    cube_marker.pose.position.x = float(tvec[0])
                    cube_marker.pose.position.y = float(tvec[1])
                    cube_marker.pose.position.z = float(tvec[2])

                    cube_marker.pose.orientation.x = quat[0]
                    cube_marker.pose.orientation.y = quat[1]
                    cube_marker.pose.orientation.z = quat[2]
                    cube_marker.pose.orientation.w = quat[3]

                    cube_marker.scale.x = marker_size
                    cube_marker.scale.y = marker_size
                    cube_marker.scale.z = 0.01  # Тонкий куб

                    cube_marker.color.r = 1.0
                    cube_marker.color.g = 1.0
                    cube_marker.color.b = 0.0
                    cube_marker.color.a = 0.8

                    markers_list.append(cube_marker)

                    axis_length = marker_size * 1.5
                    axis_width = 0.005

                    x_axis = Marker()
                    x_axis.header = header
                    x_axis.header.frame_id = 'camera_link'
                    x_axis.ns = "aruco_axes"
                    x_axis.id = marker_id * 10 + 1
                    x_axis.type = Marker.ARROW
                    x_axis.action = Marker.ADD

                    x_axis.pose.position.x = float(tvec[0])
                    x_axis.pose.position.y = float(tvec[1])
                    x_axis.pose.position.z = float(tvec[2])
                    x_axis.pose.orientation = cube_marker.pose.orientation

                    x_axis.scale.x = axis_length
                    x_axis.scale.y = axis_width
                    x_axis.scale.z = axis_width

                    x_axis.color.r = 1.0
                    x_axis.color.g = 0.0
                    x_axis.color.b = 0.0
                    x_axis.color.a = 1.0

                    markers_list.append(x_axis)

                    y_axis = Marker()
                    y_axis.header = header
                    y_axis.header.frame_id = 'camera_link'
                    y_axis.ns = "aruco_axes"
                    y_axis.id = marker_id * 10 + 2
                    y_axis.type = Marker.ARROW
                    y_axis.action = Marker.ADD

                    y_axis.pose.position.x = float(tvec[0])
                    y_axis.pose.position.y = float(tvec[1])
                    y_axis.pose.position.z = float(tvec[2])

                    y_rotation = tf_transformations.quaternion_from_euler(0, 0, np.pi/2)
                    marker_quat = [quat[0], quat[1], quat[2], quat[3]]
                    combined_quat = tf_transformations.quaternion_multiply(marker_quat, y_rotation)

                    y_axis.pose.orientation.x = combined_quat[0]
                    y_axis.pose.orientation.y = combined_quat[1]
                    y_axis.pose.orientation.z = combined_quat[2]
                    y_axis.pose.orientation.w = combined_quat[3]

                    y_axis.scale.x = axis_length
                    y_axis.scale.y = axis_width
                    y_axis.scale.z = axis_width

                    y_axis.color.r = 0.0
                    y_axis.color.g = 1.0
                    y_axis.color.b = 0.0
                    y_axis.color.a = 1.0

                    markers_list.append(y_axis)

                    z_axis = Marker()
                    z_axis.header = header
                    z_axis.header.frame_id = 'camera_link'
                    z_axis.ns = "aruco_axes"
                    z_axis.id = marker_id * 10 + 3
                    z_axis.type = Marker.ARROW
                    z_axis.action = Marker.ADD

                    z_axis.pose.position.x = float(tvec[0])
                    z_axis.pose.position.y = float(tvec[1])
                    z_axis.pose.position.z = float(tvec[2])

                    z_rotation = tf_transformations.quaternion_from_euler(0, -np.pi/2, 0)
                    combined_quat = tf_transformations.quaternion_multiply(marker_quat, z_rotation)

                    z_axis.pose.orientation.x = combined_quat[0]
                    z_axis.pose.orientation.y = combined_quat[1]
                    z_axis.pose.orientation.z = combined_quat[2]
                    z_axis.pose.orientation.w = combined_quat[3]

                    z_axis.scale.x = axis_length
                    z_axis.scale.y = axis_width
                    z_axis.scale.z = axis_width

                    z_axis.color.r = 0.0
                    z_axis.color.g = 0.0
                    z_axis.color.b = 1.0
                    z_axis.color.a = 1.0

                    markers_list.append(z_axis)

                    text_marker = Marker()
                    text_marker.header = header
                    text_marker.header.frame_id = 'camera_link'
                    text_marker.ns = "aruco_text"
                    text_marker.id = marker_id * 10 + 4
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD

                    text_marker.pose.position.x = float(tvec[0])
                    text_marker.pose.position.y = float(tvec[1])
                    text_marker.pose.position.z = float(tvec[2]) + axis_length

                    text_marker.scale.z = 0.02

                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    text_marker.color.a = 1.0

                    text_marker.text = f"ArUco {marker_id}"

                    markers_list.append(text_marker)

                marker_array.markers = markers_list

                self.marker_pub.publish(marker_array)
                self.get_logger().info(f'Published visualization for {len(ids)} ArUco markers')


def main(args=None):
    rclpy.init(args=args)
    
    slam_node = SLAMNode()
    
    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
