"""
This file provides a Particle class and a MonteCarloLocalization Class
"""

import copy as copy
import random
from math import cos, pi, sin

import numpy as np
from cost_map import CostMap
from robot import LiDARMeasurement, Pose3D


class Particle:
    """class to handle the particle of the monte carlo localization algorithm
    ATTRIBUTES:
        self.pose: (robot.pose.Pose3D)  pose of the particle (x, z, theta)
        self.weight: (number) weight of the particle
    """

    pose: Pose3D
    weight: float

    def __init__(self, pose: Pose3D = None, weight: float = 0.0):  # type: ignore
        """constructor of the class
        :param pose: (robot.pose.Pose3D)  pose of the particle (x, z, theta)
        :param weight: (number) weight of the particle
        """
        if pose is None:
            pose = Pose3D()
        self.pose = copy.copy(pose)  # (robot.pose.Pose3D)  pose of the particle (x, z, theta) in meter
        self.weight = weight  # (number)   weight of the particle

    def __str__(self):
        """to display a Particle as a string"""
        return f'{self.pose} : {self.weight}'

    def __eq__(self, other):
        """
        Overwrite the == operator
        :param other: (Particle) the Particle to be compared with
        """
        if other is None:
            return False
        else:
            return self.pose == other.pose

    def __lt__(self, other):
        """ " """
        return self.weight < other.weight

    def __le__(self, other):
        """ " """
        return self.weight <= other.weight

    def __gt__(self, other):
        """ " """
        return self.weight > other.weight

    def __ge__(self, other):
        """ " """
        return self.weight >= other.weight


class MonteCarloLocalization:
    """class to handle the Monte Carlo Localization
    ATTRIBUTES:
        self.particles: (list of Particle) the particles
        self.nb_particles: (int) number of particles
        self.max_weight: (number) current maximal weight of a particle
        self.id_best_particle: (int) the index of the best particle
    """

    particles: list[Particle]
    nb_particles: int
    max_weight: float
    id_best_particle: int

    def __init__(self):
        """constructor of the class"""
        self.particles = []  # List of the particles
        self.nb_particles = 0  # number of particles
        self.max_weight = 0.0  # current maximal weight of a particle
        self.id_best_particle = None  # type: ignore # the current best particle (should be an index in the particles list)

    def init_particles(self, cost_map: CostMap, number_of_particles: int):
        """function that initialises the particles
        the cost map is needed because we do not want to put particles in
        a non obstacle free cell
        the particles are uniformly spread over the map
        :param cost_map: (environment.cost_map.CostMap) the cost map
        :param number_of_particles: (int) the number of particles
        """
        self.nb_particles = number_of_particles
        self.particles = []
        self.max_weight = 0

        # Генерируем частицы в свободных ячейках карты
        # Карта имеет координаты от -10 до +10
        for _ in range(number_of_particles):
            # Находим случайную свободную ячейку
            while True:
                # Случайные координаты в пределах карты (-10, 10)
                x = random.uniform(-10.0, 10.0)
                z = random.uniform(-10.0, 10.0)

                # Проверяем, что ячейка свободна
                if cost_map.cell_is_empty(x, z):
                    # Случайная ориентация
                    theta = random.uniform(0, 2 * pi)

                    # Создаем частицу с равномерным весом
                    pose = Pose3D(x, z, theta)
                    particle = Particle(pose, 1.0 / number_of_particles)
                    self.particles.append(particle)
                    break

    def __str__(self):
        """
        To display a MonteCarloLocalization as a string (debug propose)
        """
        msg = ''
        for p in self.particles:
            msg += str(p) + '\n'
        return msg

    def evaluate_particles(self, cost_map: CostMap, measurements: list[LiDARMeasurement]):
        """function that update the particles weight according to the measurements
        and the cost map
        note that the max weight and the best particle need to be computed and
        the weight needs to be normalized (between 0 and 1)
        :param cost_map: (environment.cost_map.CostMap) the cost map
        :param measurements: (list of robot.lidar.LiDARMeasurement) the measurements
        """
        # Оцениваем стоимость для каждой частицы
        total_weight = 0.0
        self.max_weight = 0.0
        self.id_best_particle = 0

        costs = []
        for i, particle in enumerate(self.particles):
            # Вычисляем стоимость измерений для текущей частицы
            cost = cost_map.evaluate_cost(particle.pose, measurements)
            costs.append(cost)

        # Находим минимальную стоимость для нормализации
        min_cost = min(costs)

        for i, (particle, cost) in enumerate(zip(self.particles, costs)):
            # Преобразуем стоимость в вес с более агрессивной дискриминацией
            if cost == float('inf') or cost > 500:
                particle.weight = 0.0
            else:
                # Используем разность от минимальной стоимости для лучшей дискриминации
                cost_diff = cost - min_cost
                if cost_diff < 0.1:  # Очень близко к лучшей стоимости
                    particle.weight = 1.0
                elif cost_diff < 1.0:  # Близко к лучшей
                    particle.weight = np.exp(-cost_diff * 5.0)
                elif cost_diff < 10.0:  # Умеренно близко
                    particle.weight = np.exp(-cost_diff * 2.0)
                else:  # Далеко от лучшей
                    particle.weight = np.exp(-cost_diff * 0.5)

            total_weight += particle.weight

            # Обновляем максимальный вес и лучшую частицу
            if particle.weight > self.max_weight:
                self.max_weight = particle.weight
                self.id_best_particle = i

        # НЕ нормализуем веса - оставляем их абсолютные значения для анализа
        # Нормализация происходит только при передискретизации
        if total_weight > 0:
            # Сохраняем абсолютные веса для диагностики
            pass
        else:
            # Если все веса равны 0, присваиваем равномерные веса
            for particle in self.particles:
                particle.weight = 1.0 / len(self.particles)

    def re_sampling(self, sigma_xz: float = 0.05, sigma_theta: float = 1 * pi / 180):
        """function that re-sample the particles around the best ones
        note: use a gaussian distribution around the best particles with
        sigma for position : 0.05
        sigma for orientation : 1 deg
        """
        if not self.particles:
            return

        # Создаем новый набор частиц
        new_particles = []

        # Более агрессивная стратегия: используем top 30% частиц
        sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)
        top_count = max(3, len(sorted_particles) // 3)  # Минимум 3 частицы
        top_particles = sorted_particles[:top_count]

        weights = [p.weight for p in top_particles]
        total_weight = sum(weights)

        if total_weight <= 0:
            # Если все веса равны 0, используем равномерное распределение
            for _ in range(self.nb_particles):
                selected_particle = random.choice(self.particles)
                new_x = selected_particle.pose.x + random.gauss(0, sigma_xz)
                new_z = selected_particle.pose.z + random.gauss(0, sigma_xz)
                new_theta = selected_particle.pose.theta + random.gauss(0, sigma_theta)

                while new_theta > pi:
                    new_theta -= 2 * pi
                while new_theta < -pi:
                    new_theta += 2 * pi

                new_pose = Pose3D(new_x, new_z, new_theta)
                new_particle = Particle(new_pose, 1.0 / self.nb_particles)
                new_particles.append(new_particle)
        else:
            # Нормализуем веса для передискретизации
            normalized_weights = [w / total_weight for w in weights]

            for _ in range(self.nb_particles):
                # Выбираем частицу на основе весов
                r = random.random()
                cumsum = 0.0
                selected_particle = top_particles[0]

                for i, particle in enumerate(top_particles):
                    cumsum += normalized_weights[i]
                    if r <= cumsum:
                        selected_particle = particle
                        break

                # Добавляем шум (меньше для лучшей конвергенции)
                new_x = selected_particle.pose.x + random.gauss(0, sigma_xz * 0.5)
                new_z = selected_particle.pose.z + random.gauss(0, sigma_xz * 0.5)
                new_theta = selected_particle.pose.theta + random.gauss(0, sigma_theta * 0.5)

                while new_theta > pi:
                    new_theta -= 2 * pi
                while new_theta < -pi:
                    new_theta += 2 * pi

                new_pose = Pose3D(new_x, new_z, new_theta)
                new_particle = Particle(new_pose, 1.0 / self.nb_particles)
                new_particles.append(new_particle)

        self.particles = new_particles

    def estimate_from_odometry(self, odo_delta_dst: float, odo_delta_theta: float):
        """function that update the position and orientation of all the particles
        according to the odometry data
        PARAMETERS:
            odo_delta_dst: (number) the distance delta
            odo_delta_theta: (number) the orientation delta
        """
        for particle in self.particles:
            # Обновляем ориентацию
            particle.pose.theta += odo_delta_theta

            # Нормализуем угол
            while particle.pose.theta > pi:
                particle.pose.theta -= 2 * pi
            while particle.pose.theta < -pi:
                particle.pose.theta += 2 * pi

            # Обновляем позицию с учетом ориентации
            particle.pose.x += odo_delta_dst * cos(particle.pose.theta)
            particle.pose.z += odo_delta_dst * sin(particle.pose.theta)

            # Добавляем шум для модели движения
            particle.pose.x += random.gauss(0, 0.02)  # 2см шума по x
            particle.pose.z += random.gauss(0, 0.02)  # 2см шума по z
            particle.pose.theta += random.gauss(0, 0.05)  # ~3 градуса шума по углу

    def add_random_particles(self, cost_map: CostMap, percent: float):
        """function that modifies "percent" percent of particles by initializing them
        randomly
        percent is a value between 0 and 100
        PARAMETERS:
            cost_map: (CostMap) the cost map
            percent: (int) the percent (from 0 to 100)
        """
        num_random = int(self.nb_particles * percent / 100.0)

        # Заменяем случайно выбранные частицы новыми случайными
        indices_to_replace = random.sample(range(self.nb_particles), num_random)

        for i in indices_to_replace:
            # Генерируем новую случайную частицу в свободной ячейке
            attempts = 0
            while attempts < 100:  # Ограничиваем количество попыток
                x = random.uniform(-10.0, 10.0)
                z = random.uniform(-10.0, 10.0)

                if cost_map.cell_is_empty(x, z):
                    theta = random.uniform(0, 2 * pi)
                    pose = Pose3D(x, z, theta)
                    self.particles[i] = Particle(pose, 1.0 / self.nb_particles)
                    break

                attempts += 1
