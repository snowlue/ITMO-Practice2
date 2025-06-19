import math
import numpy as np
from sklearn.neighbors import KDTree

import pickle

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
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.map_pub = self.create_publisher(PointCloud2, '/icp/map', 10)

        self.prev_pcd = None
        self.global_map = np.empty((0, 2))
        self.acum_R = np.eye(2)
        self.acum_t = np.zeros(2)

        self.get_logger().info('ICP node initialized, waiting for /scan…')

    def publish_map(self, header):
        if self.global_map.shape[0] > 0:
            header.frame_id = 'map'
            fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                      PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                      PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
            points_list = []
            for i in range(self.global_map.shape[0]):
                points_list.append((float(self.global_map[i, 0]), float(self.global_map[i, 1]), 0.0))
            cloud_msg = pc2.create_cloud(header, fields, points_list)
            self.map_pub.publish(cloud_msg)

    def icp(self, S_move, S_fix, max_iterations=20, tolerance=1e-6, max_distance=1.0):
        """
        S_move: np.ndarray, shape (m, 2) — новая точка (двумерная)
        S_fix: np.ndarray, shape (m, 2) — предыдущая точка (двумерная)
        max_distance: float — максимальная дистанция для соответствий
        
        Модифицированная версия с взаимно-однозначным соответствием точек.
        Каждой точке из одного множества соответствует ровно одна точка из другого.
        """
        src = np.copy(S_move)  # исходная (двигаемая) облако точек
        dst = np.copy(S_fix)  # целевое (фиксированное)
        
        R = np.eye(2)
        t = np.zeros(2)

        R_global = np.eye(2)
        t_global = np.zeros(2)
        try:
            tree = KDTree(dst)
        except ValueError as e:
            print(dst)
            self.get_logger().error(f"Error creating KDTree: {dst}")
            exit()
        
        for _ in range(max_iterations):
            # 1. Поиск ближайших соседей
            dist, ind = tree.query(src, k=1)
            dist = dist.ravel()
            ind = ind.ravel()
            
            # Применяем маску по дистанции
            distance_mask = dist < max_distance
            # src = src[distance_mask]
            # matched_dst = dst[ind[distance_mask]]
            
            # Фильтруем по дистанции
            filtered_src_indices = np.where(distance_mask)[0]
            filtered_dst_indices = ind[distance_mask]
            filtered_distances = dist[distance_mask]
            
            unique_correspondences = {}
            for i, (src_idx, dst_idx, distance) in enumerate(zip(filtered_src_indices, filtered_dst_indices, filtered_distances)):
                if dst_idx not in unique_correspondences or distance < unique_correspondences[dst_idx][1]:
                    unique_correspondences[dst_idx] = (src_idx, distance)
            
            # Извлекаем финальные соответствия
            final_src_indices = []
            final_dst_indices = []
            for dst_idx, (src_idx, _) in unique_correspondences.items():
                final_src_indices.append(src_idx)
                final_dst_indices.append(dst_idx)
            
            final_src_indices = np.array(final_src_indices)
            final_dst_indices = np.array(final_dst_indices)
            
            src = src[final_src_indices]
            matched_dst = dst[final_dst_indices]
            
            if len(src) == 0 or len(matched_dst) == 0:
                break

            # 2. Центроиды
            mean_dst = np.mean(matched_dst, axis=0)
            mean_src = np.mean(src, axis=0)

            # 3. Ковариационная матрица
            Cov = np.einsum('ni,nj->ij', src - mean_src, matched_dst - mean_dst)

            # 4. SVD и получение поворота
            U, _, V = np.linalg.svd(Cov)
            R = V @ U.T

            # 5. Смещение
            t = mean_dst - R @ mean_src

            # 6. Применяем трансформацию
            src_transformed = (R @ src.T).T + t

            t_global += R_global @ t
            R_global = R @ R_global

            # Проверка сходимости
            if np.linalg.norm(src_transformed - src) < tolerance:
                break

            src = src_transformed

        return t_global, R_global

    def scan_callback(self, scan: LaserScan):
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        pcd = np.vstack((xs, ys)).T

        history.append((pcd.copy()))

        if self.prev_pcd is None:
            self.prev_pcd = pcd
            self.global_map = np.copy(pcd)
            self.get_logger().info(f'Saved reference scan with {len(pcd)} points. Map size: {self.global_map.shape[0]}')
            
            header = scan.header
            self.publish_map(header)
            return

        pcd = (self.acum_R @ pcd.T).T + self.acum_t
        
        t, R = self.icp(pcd, self.global_map, max_iterations=200, max_distance=1.0)
        pcd = (R @ pcd.T).T + t

        self.acum_t += self.acum_R @ t 
        self.acum_R = self.acum_R @ R

        self.global_map = np.vstack((self.global_map, pcd))

        header = scan.header
        self.publish_map(header)

        self.prev_pcd = pcd
        
        self.get_logger().info(f'Added {len(pcd)} points to map. Total map size: {self.global_map.shape[0]} points')


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
        pickle.dump(history, open('icp_history.pkl', 'wb'))
    else:
        print("No history to save.")

history = []


if __name__ == '__main__':
    main()