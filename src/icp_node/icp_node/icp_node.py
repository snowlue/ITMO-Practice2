import math
import numpy as np

import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import LaserScan
from rclpy.node import Node


class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_node')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/icp/cloud', 10)

        self.prev_pcd = None
        self.get_logger().info('ICP node initialized, waiting for /scan…')

    def publish_transform(self, t, R, stamp):
        tr = TransformStamped()
        tr.header.stamp = stamp
        tr.header.frame_id = 'icp_prev'
        tr.child_frame_id = 'icp_curr'
        tr.transform.translation.x = float(t[0])
        tr.transform.translation.y = float(t[1])
        tr.transform.translation.z = 0.0
        yaw = math.atan2(R[1, 0], R[0, 0])
        tr.transform.rotation.z = math.sin(yaw / 2)
        tr.transform.rotation.w = math.cos(yaw / 2)
        self.tf_broadcaster.sendTransform(tr)

    def publish_cloud(self, pts, header):
        header.frame_id = 'icp_curr'
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
        cloud_msg = pc2.create_cloud(header, fields, [(float(x), float(y), 0.0) for x, y in pts])
        self.pc_pub.publish(cloud_msg) 

    def icp(self, S_move, S_fix, max_iterations=20, tolerance=1e-6):
        """
        S_move: np.ndarray, shape (m, 2) — новая точка (двумерная)
        S_fix: np.ndarray, shape (m, 2) — предыдущая точка (двумерная)
        """
        src = np.copy(S_move)  # исходная (двигаемая) облако точек
        dst = np.copy(S_fix)  # целевое (фиксированное)
        # t = np.eye(4)

        for _ in range(max_iterations):
            # 1. Поиск ближайших соседей
            distances = np.linalg.norm(src[:, np.newaxis, :] - dst[np.newaxis, :, :], axis=2)
            indices = np.argmin(distances, axis=1)
            matched_dst = dst[indices]

            # 2. Центроиды
            mean_src = np.mean(src, axis=0)
            mean_dst = np.mean(matched_dst, axis=0)

            # 3. Центрированные точки
            src_centered = src - mean_src
            dst_centered = matched_dst - mean_dst

            # 4. Ковариационная матрица
            Cov = src_centered.T @ dst_centered

            # 5. SVD и получение поворота
            U, _, Vt = np.linalg.svd(Cov)
            R = U @ Vt

            # Обеспечиваем правильную ориентацию (чтобы детерминант был +1)
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R = U @ Vt

            # 6. Смещение
            t = mean_dst - R @ mean_src

            # 7. Применяем трансформацию
            src_transformed = (R @ src.T).T + t

            # Проверка сходимости
            if np.linalg.norm(src_transformed - src) < tolerance:
                break

            src = src_transformed

        return t, R

    def scan_callback(self, scan: LaserScan):
        # self.get_logger().info("scan:" + str(scan))

        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        pcd = np.vstack((xs, ys)).T

        if self.prev_pcd is None:
            self.prev_pcd = pcd
            self.get_logger().info('Saved reference scan.')
            return

        T, R = self.icp(pcd, self.prev_pcd)

        header = scan.header
        self.publish_transform(T, R, header.stamp)
        self.publish_cloud(pcd, header)
        # dx, dy = T
        # yaw = math.atan2(R[1, 0], R[0, 0])  # угол поворота в плоскости XY

        # self.get_logger().info(f'ICP: Δx={dx:.3f} m, Δy={dy:.3f} m, Δyaw={math.degrees(yaw):.2f}°')

        # Обновляем эталон
        self.prev_pcd = pcd


def main(args=None):
    # try:
    #     with rclpy.init(args=args):
    #         node = DriverNode()
    #         rclpy.spin(node)
    # except (KeyboardInterrupt, ExternalShutdownException):
    #     pass
    rclpy.init(args=args)
    node = ICPNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
