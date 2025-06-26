# coding: utf-8
"""
This file provides a Pose class, and a LiDAR and a LiDARMeasurement class, and a Robot class to simulate a two wheeled mobile robot dynamic
"""

from math import cos, pi, sin

from environment import SegEnv
from geometry import Point2D, Segment2D


class Pose3D:
    """A class to handle a 3D pose (2D position and 1 orientation)
    ATTRIBUTES:
        self.x: (number in m) the x position
        self.z: (number in m) the z position
        self.theta: (number in rad) the orientation
    """

    x: float
    z: float
    theta: float

    def __init__(self, x=0, z=0, theta=0):
        """Constructor of the class
        PARAMETERS:
            x: (number in m) the x position
            z: (number in m) the z position
            theta: (number in rad) the orientation
        """
        self.x = x  # m      x position of the robot
        self.z = z  # m      z position of the robot
        self.theta = theta  # rad    orientation of the robot

    def __str__(self):
        """to print a pose 3D"""
        return f'({self.x}, {self.z}, {self.theta})'

    def __eq__(self, other):
        """
        Overwrite the == operator
        :param other: (Pose3D) the pose to be compared with
        """
        return self.x == other.x and self.z == other.z and self.theta == other.theta

    def __add__(self, other):
        """
        Overwrite the self + other operator
        :param other: (Pose3D) the pose to add
        """
        if isinstance(other, Pose3D):
            return Pose3D(self.x + other.x, self.z + other.z, self.theta + other.theta)  # type: ignore
        else:
            return Pose3D(self.x + other, self.z + other, self.theta + other)

    def __radd__(self, other):
        """
        Overwrite the other + self operator
        :param other: (Pose3D) the pose to add
        """
        return self + other

    def __sub__(self, other):
        """
        Overwrite the self - other operator
        :param other: (Pose3D) the pose to sub
        """
        if isinstance(other, Pose3D):
            return Pose3D(self.x - other.x, self.z - other.z, self.theta - other.theta)  # type: ignore
        else:
            return Pose3D(self.x - other, self.z - other, self.theta - other)

    def __rsub__(self, other):
        """
        Overwrite the other + self operator
        :param other: (Pose3D) the pose to add
        """
        return -self + other

    def __neg__(self):
        """
        Overwrite the negative operator (-self)
        """
        return Pose3D(-self.x, -self.z, -self.theta)  # type: ignore

    def __mul__(self, other):
        """
        Overwrite the self * other operator
        :param other: (Pose3D) the pose to add
        """
        if isinstance(other, Pose3D):
            return Pose3D(self.x * other.x, self.z * other.z, self.theta * other.theta)  # type: ignore
        else:
            return Pose3D(self.x * other, self.z * other, self.theta * other)

    def __rmul__(self, other):
        """
        Overwrite the other * self operator
        :param other: (Pose3D) the pose to add
        """
        return self * other


class LiDARMeasurement:
    """class to handle LiDAR measurement"""

    distance: float  # (float in m) the distance of the LiDAR measurement
    angle: float  # (float in rad) the angle of the LiDAR measurement

    def __init__(self, distance: float = 0.0, angle: float = 0.0):
        """constructor of the class
        PARAMETERS:
            distance : (number in m) the distance of the LiDAR measurement
            angle : (number in rad) the angle of the LiDAR measurement
        """
        self.distance = distance
        self.angle = angle


class LiDAR:
    """class to handle a LiDAR sensor"""

    range: float  # (number in m) the distance of the LiDAR measurement

    def __init__(self, sensor_range: float = 0):
        """Constructor of the class
        PARAMETERS:
            sensor_range: (number in m) the range of the sensor
        """
        self.range = sensor_range  # m  maximal distance of a measurement

    def init_sensor(self):
        """Function that initialize the LiDAR"""
        # variables for the model
        self.range = 5

    def get_measurements(self, pose: Pose3D, environment: SegEnv) -> list[LiDARMeasurement]:
        """Function that returns a measurement set, according to the position
        of the sensor (x, z), its orientation (theta) and the environment
        PARAMETERS:
            pose: (robot.pose.Pose3D) the pose of the sensor in the environment frame (x, z, theta)
            environment: (environment.seg_environment SegEnv) the environment the sensor is in
        RETURNS:
            list of LiDARMeasurement
        """
        measurements = []
        for step in range(-90, 91, 10):
            point_ext = Point2D(
                cos(pose.theta + step * pi / 180) * self.range + pose.x,
                sin(pose.theta + step * pi / 180) * self.range + pose.z,
            )
            distance = Point2D.distance(point_ext, Point2D(pose.x, pose.z))

            segment_sensor = Segment2D(Point2D(pose.x, pose.z), point_ext)
            for seg in environment.segments:
                result, new_inter = Segment2D.intersect(segment_sensor, seg)
                if result is True:
                    new_distance = Point2D.distance(new_inter, Point2D(pose.x, pose.z))
                    if new_distance < distance:
                        distance = new_distance
            if distance < self.range - 0.001:
                measurements.append(LiDARMeasurement(distance, step * pi / 180))
        return measurements


class Robot:
    """A class to simulate a two wheeled mobile robot
    the input corresponds to the angular wheel speed (rad/s)
    ATTRIBUTES:
        self.wheel_radius: (number in m) radius of the left wheel
        self.wheel_distance: (number in m) distance between the left and the right wheel
        self.pose: (robot.pose.Pose3D) the pose of the robot (position and orientation)
        self.d_width_wheel: (number in m) width of the wheels
    """

    wheel_radius: float
    wheel_distance: float
    pose: Pose3D
    d_width_wheel: float

    def __init__(self):
        """Constructor of the class"""
        # variables for the model
        self.wheel_radius = None  # type: ignore # m  radius of the left wheel
        self.wheel_distance = None  # type: ignore # m  distance between the left and the right wheel

        self.pose = Pose3D()

        # variable for the display
        self.d_width_wheel = None  # type: ignore # m  width of the wheels

    def __str__(self):
        """to be able to print a Robot"""
        return f'{self.pose}'

    def init_robot(self, pose):
        """Function to initialize the robot parameters
        PARAMETERS:
            pose: (robot.pose.Pose3D) the pose for the robot (position and orientation)
        """
        # variables for the model
        self.wheel_radius = 0.1  # m  radius of the left wheel
        self.wheel_distance = 0.2  # m  distance between the left and the right wheel

        self.pose = pose  # pose of the robot (x, z, theta)

        # variable for the display
        self.d_width_wheel = 0.05  # m  width of the wheels

    def f(self, _unused_x, _unused_z, theta, delta_t, u):
        """Differential equation that models the robot
        PARAMETERS:
            _unused_x: (number in m) the x position (not used)
            _unused_z: (number in m) the z position (not used)
            theta: (number in rad) the orientation
            delta_t: (number in s) the step for the discreet time
            u: (list of two number in rad/s) the wheel speed commands [left, right]
        RETURNS:
            the x' value (derivative of the x position according to the differential equation) (number)
            the z' value (derivative of the z position according to the differential equation) (number)
            the theta' value (derivative of the orientation according to the differential equation) (number)
        """

        # variable to ease the reading
        r = self.wheel_radius  # m      radius of the left wheel
        dst = self.wheel_distance  # m      distance between the left and the right wheel
        ul = u[0]  # rad/s  angular speed of the left wheel
        ur = u[1]  # rad/s  angular speed of the right wheel

        xp = (r / 2.0) * (ul + ur) * cos(theta)
        zp = (r / 2.0) * (ul + ur) * sin(theta)
        theta_p = (r / dst) * (ul - ur)

        return xp * delta_t, zp * delta_t, theta_p * delta_t

    def runge_kutta(self, delta_t, u):
        """Function that implements the runge kutta approach for estimating differential equation
        it updates the x, z and theta attribute of the object
        PARAMETERS:
            delta_t: (number in s) time step for the runge kutta
            u: (list of two number in rad/s) the wheel speed commands [left, right]
        """
        # to estimate the derivative equation
        k1xp, k1zp, k1_theta_p = self.f(self.pose.x, self.pose.z, self.pose.theta, delta_t, u)
        k2xp, k2zp, k2_theta_p = self.f(
            self.pose.x + k1xp / 2, self.pose.z + k1zp / 2, self.pose.theta + k1_theta_p / 2, delta_t, u
        )
        k3xp, k3zp, k3_theta_p = self.f(
            self.pose.x + k2xp / 2, self.pose.z + k2zp / 2, self.pose.theta + k2_theta_p / 2, delta_t, u
        )
        k4xp, k4zp, k4_theta_p = self.f(
            self.pose.x + k3xp, self.pose.z + k3zp, self.pose.theta + k3_theta_p, delta_t, u
        )

        # update of the robot's state (position and orientation)
        self.pose.x += (1 / 6.0) * (k1xp + 2 * k2xp + 2 * k3xp + k4xp)
        self.pose.z += (1 / 6.0) * (k1zp + 2 * k2zp + 2 * k3zp + k4zp)
        self.pose.theta += (1 / 6.0) * (k1_theta_p + 2 * k2_theta_p + 2 * k3_theta_p + k4_theta_p)

    def dynamics(self, delta_t, u):
        """Function to provide an interface between the model and the display
            It updates the robot's position considering that to the command U was provided for delta t seconds
        PARAMETERS:
            delta_t: (number in s) time step for the runge kutta
            u: (list of two number in rad/s) the wheel speed commands [left, right]
        """
        self.runge_kutta(delta_t, u)
