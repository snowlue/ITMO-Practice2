# slam_tutorial_py/icp_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import open3d as o3d
import math


class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_node')
        # Параметры ICP
        self.max_correspondence_distance = 0.5  # максимальное расстояние для соответствий
        self.icp_iteration = 50

        self.prev_pcd = None
        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('ICP node initialized, waiting for /scan…')

    def scan_callback(self, scan: LaserScan):
        # Конвертация LaserScan → numpy Nx2
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        # Убираем "inf" и "nan"
        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.vstack((xs, ys, np.zeros_like(xs))).T  # делаем 3D, z=0

        # Создаём Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if self.prev_pcd is None:
            # Первый скан — сохраняем как эталон
            self.prev_pcd = pcd
            self.get_logger().info('Saved reference scan.')
            return

        # ICP регистрация
        reg = o3d.pipelines.registration.registration_icp(
            pcd,
            self.prev_pcd,
            self.max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_iteration),
        )

        # Вывод результата
        T = reg.transformation  # 4×4 матрица преобразования: [R|t; 0 1]
        dx, dy = T[0, 3], T[1, 3]
        yaw = math.atan2(T[1, 0], T[0, 0])  # угол поворота в плоскости XY

        self.get_logger().info(f'ICP: Δx={dx:.3f} m, Δy={dy:.3f} m, Δyaw={math.degrees(yaw):.2f}°')

        # Обновляем эталон
        self.prev_pcd = pcd


def main(args=None):
    rclpy.init(args=args)
    node = ICPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
