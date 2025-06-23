import math
import pickle

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
import tf_transformations
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import MapMetaData, OccupancyGrid, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from sklearn.neighbors import KDTree


class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_node')
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.map_pub = self.create_publisher(PointCloud2, '/icp/map', 10)
        self.path_pub = self.create_publisher(Path, '/icp/path', 10)
        self.og_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.prev_pcd = None
        self.global_map = np.empty((0, 2))
        self.acum_R = np.eye(2)
        self.acum_t = np.zeros(2)

        self.robot_path = Path()
        self.robot_path.header.frame_id = 'map'

        self.resolution = 0.05
        self.x_min, self.y_min = -10.0, -10.0
        self.width = int(np.ceil(20.0 / self.resolution))
        self.height = int(np.ceil(20.0 / self.resolution))
        self.og = OccupancyGrid()
        self.og.header.frame_id = 'map'
        self.og.info = MapMetaData()
        self.og.info.resolution = self.resolution
        self.og.info.width = self.width
        self.og.info.height = self.height
        self.og.info.origin.position.x = self.x_min
        self.og.info.origin.position.y = self.y_min
        self.og.info.origin.orientation.w = 1.0
        self.og.data = [-1] * (self.width * self.height)

        self.robot_position = np.zeros(2)
        self.robot_orientation = 0.0

        self.get_logger().info('ICP node initialized, waiting for /scan…')

    def publish_map(self, header):
        if self.global_map.size == 0:
            return

        now = self.get_clock().now().to_msg()
        self.og.header.stamp = now

        self.og.data = [-1] * (self.width * self.height)

        rx = int((self.robot_position[0] - self.x_min) / self.resolution)
        ry = int((self.robot_position[1] - self.y_min) / self.resolution)

        if not (0 <= rx < self.width and 0 <= ry < self.height):
            self.get_logger().warn(f'Robot position ({rx}, {ry}) is outside map bounds')
            return

        if len(self.global_map.shape) != 2 or self.global_map.shape[0] == 0:
            return

        for i in range(self.global_map.shape[0]):
            x, y = self.global_map[i, 0], self.global_map[i, 1]
            grid_x = int((x - self.x_min) / self.resolution)
            grid_y = int((y - self.y_min) / self.resolution)
            if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):
                continue

            for cx, cy in self.bresenham(rx, ry, grid_x, grid_y):
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    idx = cy * self.width + cx
                    if self.og.data[idx] == -1:
                        self.og.data[idx] = 0

            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                self.og.data[grid_y * self.width + grid_x] = 100

        self.og_pub.publish(self.og)

    def bresenham(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        cells.append((x1, y1))
        return cells

    def publish_pc(self, header):
        if self.global_map.shape[0] > 0:
            header.frame_id = 'map'
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            points_list = []
            for i in range(self.global_map.shape[0]):
                points_list.append((float(self.global_map[i, 0]), float(self.global_map[i, 1]), 0.0))
            cloud_msg = pc2.create_cloud(header, fields, points_list)
            self.map_pub.publish(cloud_msg)

    def publish_path(self, header):
        self.robot_path.header = header
        self.robot_path.header.frame_id = 'map'
        self.path_pub.publish(self.robot_path)

    def add_pose_to_path(self, header):
        pose_stamped = PoseStamped()
        pose_stamped.header = header
        pose_stamped.header.frame_id = 'map'

        pose_stamped.pose.position.x = float(self.robot_position[0])
        pose_stamped.pose.position.y = float(self.robot_position[1])
        pose_stamped.pose.position.z = 0.0

        quat = tf_transformations.quaternion_from_euler(0, 0, self.robot_orientation)
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]

        poses_list = list(self.robot_path.poses)
        poses_list.append(pose_stamped)
        self.robot_path.poses = poses_list

    def publish_transform(self, header):
        transform = TransformStamped()
        transform.header = header
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'base_link'

        transform.transform.translation.x = float(self.robot_position[0])
        transform.transform.translation.y = float(self.robot_position[1])
        transform.transform.translation.z = 0.0

        quat = tf_transformations.quaternion_from_euler(0, 0, self.robot_orientation)
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(transform)

    def icp(self, S_move, S_fix, max_iterations=20, tolerance=1e-6, max_distance=1.0, threshold_iter=20):
        """
        S_move: np.ndarray, shape (m, 2) — новая точка (двумерная)
        S_fix: np.ndarray, shape (m, 2) — предыдущая точка (двумерная)
        max_distance: float — максимальная дистанция для соответствий
        """
        src = np.copy(S_move)  # исходная (двигаемая) облако точек
        dst = np.copy(S_fix)  # целевое (фиксированное)

        dst_norm = np.linalg.norm(dst, axis=1)
        dst = dst[dst_norm > 0.1]

        P = np.copy(src)

        R = np.eye(2)
        t = np.zeros(2)

        R_global = np.eye(2)
        t_global = np.zeros(2)
        try:
            tree = KDTree(dst)
        except ValueError:
            print(dst)
            self.get_logger().error(f'Error creating KDTree: {dst}')
            exit()

        for _ in range(max_iterations):
            # 1. Поиск ближайших соседей
            dist, ind = tree.query(P, k=1)
            dist = dist.ravel()
            ind = ind.ravel()

            # Применяем маску по дистанции
            distance_mask = dist < max_distance
            P = P[distance_mask]
            matched_dst = dst[ind[distance_mask]]

            # 2. Центроиды
            mean_dst = np.mean(matched_dst, axis=0)
            mean_src = np.mean(P, axis=0)

            # 3. Ковариационная матрица
            Cov = (P - mean_src).T @ (matched_dst - mean_dst)

            # 4. SVD и получение поворота
            U, _, Vh = np.linalg.svd(Cov)
            R = Vh.T @ U.T

            # 5. Смещение
            t = mean_dst - R @ mean_src

            # 6. Применяем трансформацию
            P = (R @ P.T).T + t

            t_global += t
            R_global = R @ R_global

            # Проверка сходимости
            # if np.linalg.norm(P - matched_dst) < tolerance:
            #     break

        return src, t_global, R_global

    def scan_callback(self, scan: LaserScan):
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        pcd = np.vstack((xs, ys)).T

        # history.append((pcd.copy()))

        if self.prev_pcd is None:
            self.prev_pcd = pcd
            self.global_map = np.copy(pcd)

            self.robot_position = np.zeros(2)
            self.robot_orientation = 0.0

            self.get_logger().info(f'Saved reference scan with {len(pcd)} points. Map size: {self.global_map.shape[0]}')

            header = scan.header
            self.publish_transform(header)
            self.add_pose_to_path(header)
            self.publish_path(header)
            self.publish_map(header)
            self.publish_pc(header)
            return

        pcd_norm = np.linalg.norm(pcd, axis=1)
        pcd = pcd[pcd_norm > 0.1]

        pcd = (self.acum_R @ pcd.T).T + self.acum_t

        pcd, t, R = self.icp(pcd, self.global_map, max_iterations=100, max_distance=1.0)
        pcd = (R @ pcd.T).T + t

        self.acum_t += t
        self.acum_R = self.acum_R @ R

        self.global_map = np.vstack((self.global_map, pcd))

        self.robot_position = self.acum_t.copy()
        self.robot_orientation = math.atan2(self.acum_R[1, 0], self.acum_R[0, 0])

        self.prev_pcd = pcd

        self.get_logger().info(
            f'Added {len(pcd)} points to map. Total map size: {self.global_map.shape[0]} points. Robot pos: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}), orientation: {math.degrees(self.robot_orientation):.1f}°'
        )

        header = scan.header
        self.publish_transform(header)
        self.add_pose_to_path(header)
        self.publish_path(header)
        self.publish_map(header)
        self.publish_pc(header)


def main(args=None):
    rclpy.init(args=args)
    node = ICPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        on_shutdown()


def on_shutdown():
    if history:
        ...
        # pickle.dump(history, open('icp_history_rotation.pkl', 'wb'))
        # pickle.dump(history, open('map.pkl', 'wb'))
    else:
        print('No history to save.')


history = []


if __name__ == '__main__':
    main()
