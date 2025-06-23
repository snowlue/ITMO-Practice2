import socket
import struct
import threading
import time

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from transforms3d.euler import euler2quat


class DriverNode(Node):
    def __init__(self):
        super().__init__('driver_node')
        self.outputSignals: list[float] = [0, 0, 0, 0]
        self.publisher = self.create_publisher(Odometry, '/odom', 10)
        self.subscribtion = self.create_subscription(Twist, '/cmd_vel', self.listener_callback, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.get_logger().info('Wait for connecting')
        self.s.connect(('ev3dev.local', 5553))
        self.get_logger().info('Connected')
        pubThr = threading.Thread(target=self.send_data, args=(self.outputSignals, self.s))
        subThr = threading.Thread(target=self.receive_data, args=(self.s,))
        pubThr.start()
        subThr.start()

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

        self.tf_broadcaster.sendTransform(transform)

    def listener_callback(self, msg: Twist):
        self.outputSignals[0] = msg.linear.x
        self.outputSignals[1] = msg.angular.z

    def receive_data(self, s: socket.socket):
        T = 1 / 100

        while True:
            st = time.time()

            chunk = s.recv(12)
            if chunk == b'':
                raise RuntimeError('socket connection broken')

            # receive linear and angular speed
            x, y, th = struct.unpack('>fff', chunk)
            self.publish_info(x, y, th)

            et = time.time()
            dt = T - (et - st)
            if dt > 0:
                time.sleep(dt)

    def send_data(self, outputData: list, s: socket.socket):
        T = 1 / 20

        while True:
            st = time.time()

            sent = s.send(struct.pack('>ff', outputData[0], outputData[1]))
            if sent == 0:
                raise RuntimeError('socket connection broken')

            et = time.time()
            dt = T - (et - st)
            if dt > 0:
                time.sleep(dt)


def main(args=None):
    # try:
    #     with rclpy.init(args=args):
    #         node = DriverNode()
    #         rclpy.spin(node)
    # except (KeyboardInterrupt, ExternalShutdownException):
    #     pass
    rclpy.init(args=args)
    node = DriverNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
