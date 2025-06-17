import numpy as np

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from transforms3d.euler import euler2quat


class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_node')
        # self.publisher = self.create_publisher(Odometry, '/odom', 10)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

    def icp(self, S_move, S_fix, max_iterations=20, tolerance=1e-6):
        """
        S_move: np.ndarray, shape (m, 2) — новая точка (двумерная)
        S_fix: np.ndarray, shape (m, 2) — предыдущая точка (двумерная)
        """
        src = np.copy(S_move)  # исходная (двигаемая) облако точек
        dst = np.copy(S_fix)  # целевое (фиксированное)

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

        return src, R, t

    def publish_info(self, x, y, th):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        # q = tf_transformations.quaternion_from_euler(0, 0, th)
        w, x, y, z = euler2quat(0, 0, th)
        q = [x, y, z, w]
        msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.publisher.publish(msg)

        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'

        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0

        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]


    def scan_callback(self, msg: Twist):
        ...
        

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
