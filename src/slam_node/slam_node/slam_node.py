import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R


class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        
        self.bridge = CvBridge()
        
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_rect_color', self.image_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10
        )
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        self.marker_size = 0.093
        
        # SLAM карта - хранение трансформаций маркеров
        self.marker_transforms = {}  # marker_id -> T_world_marker (4x4 matrix)
        self.global_points = []      # накопленные глобальные точки
        self.reference_marker_id = None  # ID опорного маркера
        
        # TF broadcasters
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.get_logger().info('SLAM Node initialized, waiting for camera info and data...')
        self.pc_pub = self.create_publisher(PointCloud2, '/slam/keyframe_cloud', 10)
    
    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info('Camera info received')
    
    def image_callback(self, msg):
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
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                # 4. Оценка позы маркеров
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                )

                new_markers_found = False
                detected_markers = {}
                
                # Построить матрицы трансформации для всех детектированных маркеров
                for i in range(len(ids)):
                    marker_id = int(ids[i][0])
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]
                    
                    # Построить трансформацию T_camera←marker (ArUco дает позицию маркера в камере)
                    T_camera_marker = self.build_homogeneous_transform(rvec, tvec)
                    detected_markers[marker_id] = T_camera_marker
                    
                    self.get_logger().info(f'Detected ArUco marker {marker_id} at position: x={tvec[0]:.3f}, y={tvec[1]:.3f}, z={tvec[2]:.3f}')

                # 5. Определение мировых трансформов
                if len(self.marker_transforms) == 0:
                    # Карта пуста - первый маркер становится опорным в начале координат
                    first_marker_id = list(detected_markers.keys())[0]
                    
                    # Устанавливаем первый маркер в начале координат мировой системы
                    # T_world←marker = I (единичная матрица)
                    T_world_marker0 = np.eye(4)
                    
                    self.marker_transforms[first_marker_id] = T_world_marker0
                    self.reference_marker_id = first_marker_id
                    new_markers_found = True
                    
                    self.get_logger().info(f'Initialized map with reference marker {first_marker_id} at world origin')
                else:
                    # Карта существует - обрабатываем новые маркеры
                    reference_marker_in_view = None
                    T_camera_ref = None
                    
                    # Найти наиболее стабильный опорный маркер среди детектированных
                    # Предпочитаем маркер с наименьшим ID для консистентности
                    known_markers = []
                    for marker_id in detected_markers.keys():
                        if marker_id in self.marker_transforms:
                            known_markers.append(marker_id)
                    
                    if known_markers:
                        # Выбрать маркер с наименьшим ID для стабильности
                        reference_marker_in_view = min(known_markers)
                        T_camera_ref = detected_markers[reference_marker_in_view]
                        
                        # Есть опорный маркер в кадре - можем добавлять новые
                        T_world_ref = self.marker_transforms[reference_marker_in_view]
                        
                        for marker_id, T_camera_marker in detected_markers.items():
                            if marker_id not in self.marker_transforms:
                                # Новый маркер: вычисляем его позицию в мировых координатах
                                # T_world←M_new = T_world←M_ref + относительная позиция между маркерами
                                
                                # Относительная трансформация: T_ref←new = T_camera←ref * T_camera←new^-1
                                T_camera_ref_inv = self.invert_transform(T_camera_ref)
                                T_ref_new = T_camera_ref_inv @ T_camera_marker
                                
                                # Глобальная позиция: T_world←new = T_world←ref * T_ref←new
                                T_world_marker = T_world_ref @ T_ref_new
                                
                                self.marker_transforms[marker_id] = T_world_marker
                                new_markers_found = True
                                
                                self.get_logger().info(f'Added new marker {marker_id} to map using reference {reference_marker_in_view}')

                # 6. Преобразование точек в глобальные координаты
                current_points = []
                
                # Локальные координаты углов маркера
                half = self.marker_size / 2.0
                local_corners = np.array([
                    [-half,  half, 0, 1],
                    [ half,  half, 0, 1],
                    [ half, -half, 0, 1],
                    [-half, -half, 0, 1]
                ])
                local_center = np.array([0, 0, 0, 1])
                
                for marker_id in detected_markers.keys():
                    if marker_id in self.marker_transforms:
                        T_world_marker = self.marker_transforms[marker_id]
                        
                        # Преобразовать углы маркера в глобальные координаты
                        for corner in local_corners:
                            p_world = T_world_marker @ corner
                            current_points.append([p_world[0], p_world[1], p_world[2]])
                        
                        # Преобразовать центр маркера в глобальные координаты
                        c_world = T_world_marker @ local_center
                        current_points.append([c_world[0], c_world[1], c_world[2]])

                # Добавить новые точки к глобальному облаку при обнаружении новых маркеров
                if new_markers_found:
                    self.global_points.extend(current_points)
                    self.publish_static_transforms()
                
                # Всегда публиковать облако точек всех известных маркеров
                all_points = []
                for marker_id in self.marker_transforms.keys():
                    T_world_marker = self.marker_transforms[marker_id]
                    
                    # Добавить углы маркера
                    for corner in local_corners:
                        p_world = T_world_marker @ corner
                        all_points.append([p_world[0], p_world[1], p_world[2]])
                    
                    # Добавить центр маркера
                    c_world = T_world_marker @ local_center
                    all_points.append([c_world[0], c_world[1], c_world[2]])

                # 7. Публикация облака точек
                if len(all_points) > 0:
                    # Создать header с world frame
                    world_header = header
                    world_header.frame_id = 'world'
                    
                    cloud_msg = pc2.create_cloud_xyz32(world_header, all_points)
                    self.pc_pub.publish(cloud_msg)

                # 8. Локализация камеры в готовой карте
                if len(detected_markers) > 0:
                    # Найти наиболее надежный известный маркер для вычисления позиции камеры
                    # Предпочитаем маркер с наименьшим ID для консистентности
                    camera_localized = False
                    known_markers_sorted = sorted([mid for mid in detected_markers.keys() 
                                                 if mid in self.marker_transforms])
                    
                    for marker_id in known_markers_sorted:
                        T_camera_marker = detected_markers[marker_id]
                        T_world_marker = self.marker_transforms[marker_id]
                        
                        # Позиция камеры в мировых координатах:
                        # T_world←camera = T_world←marker * T_marker←camera
                        # где T_marker←camera = T_camera←marker^-1
                        T_marker_camera = self.invert_transform(T_camera_marker)
                        T_world_camera = T_world_marker @ T_marker_camera
                        
                        self.publish_camera_transform(T_world_camera)
                        camera_localized = True
                        break
                    
                    if not camera_localized:
                        self.get_logger().warn('Could not localize camera - no known markers detected')

                self.get_logger().info(f'Processed {len(ids)} ArUco markers, map contains {len(self.marker_transforms)} markers')
    
    def build_homogeneous_transform(self, rvec, tvec):
        """Построить однородную матрицу трансформации 4x4 из rvec и tvec"""
        R_mat, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = tvec
        return T
    
    def camera_to_marker_transform(self, rvec, tvec):
        """Построить трансформацию T_marker←camera из ArUco данных"""
        # ArUco возвращает позицию и ориентацию маркера в системе координат камеры
        # Нам нужна трансформация из камеры в маркер
        R_mat, _ = cv2.Rodrigues(rvec)
        
        # T_marker←camera = [R^T, -R^T * t; 0, 1]
        T = np.eye(4)
        T[:3, :3] = R_mat.T  # Транспонированная матрица поворота
        T[:3, 3] = -R_mat.T @ tvec  # Инвертированное смещение
        return T
    
    def rotation_matrix_to_quaternion(self, R_mat):
        """Конвертировать матрицу поворота в кватернион используя scipy"""
        try:
            rotation = R.from_matrix(R_mat)
            quat = rotation.as_quat()  # [x, y, z, w]
            return quat[3], quat[0], quat[1], quat[2]  # [w, x, y, z]
        except Exception:
            # Fallback к единичному кватерниону
            return 1.0, 0.0, 0.0, 0.0
    
    def invert_transform(self, T):
        """Инвертировать однородную матрицу трансформации"""
        T_inv = np.eye(4)
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv
    
    def publish_static_transforms(self):
        """Публиковать статические трансформы всех известных маркеров"""
        for marker_id, T_world_marker in self.marker_transforms.items():
            self.publish_marker_transform(marker_id, T_world_marker)
    
    def publish_marker_transform(self, marker_id, T_world_marker):
        """Публиковать статический трансформ маркера"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = f'marker_{marker_id}'
        
        # Извлечь позицию
        t.transform.translation.x = float(T_world_marker[0, 3])
        t.transform.translation.y = float(T_world_marker[1, 3])
        t.transform.translation.z = float(T_world_marker[2, 3])
        
        # Конвертировать матрицу поворота в кватернион
        R_mat = T_world_marker[:3, :3]
        w, x, y, z = self.rotation_matrix_to_quaternion(R_mat)
        t.transform.rotation.w = float(w)
        t.transform.rotation.x = float(x)
        t.transform.rotation.y = float(y)
        t.transform.rotation.z = float(z)
        
        self.static_tf_broadcaster.sendTransform(t)
    
    def publish_camera_transform(self, T_world_camera):
        """Публиковать динамический трансформ камеры"""
        # Убран фильтр стабильности - публикуем все трансформации
        T_smoothed = T_world_camera
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'camera'
        
        t.transform.translation.x = float(T_smoothed[0, 3])
        t.transform.translation.y = float(T_smoothed[1, 3])
        t.transform.translation.z = float(T_smoothed[2, 3])
        
        # Конвертировать в кватернион
        R_mat = T_smoothed[:3, :3]
        w, x, y, z = self.rotation_matrix_to_quaternion(R_mat)
        t.transform.rotation.w = float(w)
        t.transform.rotation.x = float(x)
        t.transform.rotation.y = float(y)
        t.transform.rotation.z = float(z)
        
        self.dynamic_tf_broadcaster.sendTransform(t)


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
