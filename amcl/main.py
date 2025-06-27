#!/usr/bin/env python3
"""
Улучшенная версия AMCL с более строгой локализацией
"""

import copy
import pickle
from math import atan2, pi, sqrt

import matplotlib.pyplot as plt
import numpy as np
from cost_map import CostMap
from environment import GridCell, GridMap
from monte_carlo import MonteCarloLocalization
from robot import LiDARMeasurement, Pose3D


def load_data():
    with open('pickles/laserscans.pkl', 'rb') as f:
        laserscans = pickle.load(f)
    with open('pickles/path.pkl', 'rb') as f:
        path_data = pickle.load(f)
    with open('pickles/map.pkl', 'rb') as f:
        map_data = pickle.load(f)
    return laserscans, path_data, map_data


def create_grid_map_from_data(map_data):
    grid_map = GridMap()
    grid_map.width = 20.0
    grid_map.height = 20.0
    grid_map.nb_cell_x = 400
    grid_map.nb_cell_z = 400
    grid_map.size_x = grid_map.width / grid_map.nb_cell_x
    grid_map.size_z = grid_map.height / grid_map.nb_cell_z

    map_2d = np.array(map_data).reshape(400, 400)

    grid_map.cells = []
    for z in range(grid_map.nb_cell_z):
        grid_map.cells.append([])
        for x in range(grid_map.nb_cell_x):
            cell = GridCell(x, z)
            map_value = map_2d[z, x]
            if map_value == 100:
                cell.val = 1.0
            else:
                cell.val = 0.0
            grid_map.cells[z].append(cell)

    return grid_map, map_2d


def convert_laserscan_to_measurements(laserscan_points):
    measurements = []
    for point in laserscan_points:
        x, y = point[0], point[1]
        distance = sqrt(x * x + y * y)
        angle = atan2(y, x)
        if distance > 0.1:
            measurements.append(LiDARMeasurement(distance, angle))
    return measurements


def extract_pose_from_stamped(pose_stamped):
    pos = pose_stamped.pose.position
    orientation = pose_stamped.pose.orientation
    yaw = 2 * atan2(orientation.z, orientation.w)
    return Pose3D(pos.x, pos.y, yaw)


def compute_odometry_delta(prev_pose, curr_pose):
    dx = curr_pose.x - prev_pose.x
    dz = curr_pose.z - prev_pose.z

    delta_dist = sqrt(dx * dx + dz * dz)
    delta_theta = curr_pose.theta - prev_pose.theta

    while delta_theta > pi:
        delta_theta -= 2 * pi
    while delta_theta < -pi:
        delta_theta += 2 * pi

    return delta_dist, delta_theta


def initialize_particles_at_origin(mcl, cost_map, num_particles=100):
    """Инициализируем частицы с заданной стартовой позицией (0, 0)"""
    print('Инициализируем частицы с стартовой позицией (0, 0)...')

    mcl.nb_particles = num_particles
    mcl.particles = []
    mcl.max_weight = 0

    import random

    from monte_carlo import Particle

    # Проверяем, что точка (0, 0) доступна
    if not cost_map.cell_is_empty(0.0, 0.0):
        print('ВНИМАНИЕ: Стартовая позиция (0, 0) недоступна! Ищем ближайшую свободную позицию...')
        # Ищем ближайшую свободную позицию к (0, 0)
        start_x, start_z = 0.0, 0.0
        for radius in [0.1, 0.2, 0.5, 1.0]:
            for angle in [0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4]:
                test_x = radius * np.cos(angle)
                test_z = radius * np.sin(angle)
                if cost_map.cell_is_empty(test_x, test_z):
                    start_x, start_z = test_x, test_z
                    print(f'Найдена свободная стартовая позиция: ({start_x:.3f}, {start_z:.3f})')
                    break
            if cost_map.cell_is_empty(start_x, start_z):
                break
    else:
        start_x, start_z = 0.0, 0.0
        print('Стартовая позиция (0, 0) доступна')

    # Инициализируем все частицы в стартовой позиции с разными ориентациями
    for i in range(num_particles):
        # Добавляем небольшой шум к позиции (±5см для реалистичности)
        x = start_x + random.gauss(0, 0.05)
        z = start_z + random.gauss(0, 0.05)

        # Случайная ориентация
        theta = random.uniform(0, 2 * pi)

        # Проверяем что позиция с шумом тоже свободна
        if cost_map.cell_is_empty(x, z):
            pose = Pose3D(x, z, theta)
            particle = Particle(pose, 1.0 / num_particles)
            mcl.particles.append(particle)
        else:
            # Если позиция с шумом занята, используем точную стартовую позицию
            pose = Pose3D(start_x, start_z, theta)
            particle = Particle(pose, 1.0 / num_particles)
            mcl.particles.append(particle)

    print(f'Инициализировано {len(mcl.particles)} частиц в стартовой позиции ({start_x:.3f}, {start_z:.3f})')
    return len(mcl.particles) > 0


def run_amcl():
    """Запускает улучшенный AMCL алгоритм"""
    print('Запуск улучшенного AMCL...')

    # Загружаем данные
    laserscans, path_data, map_data = load_data()
    grid_map, map_2d = create_grid_map_from_data(map_data)

    # Создаем карту стоимости
    cost_map = CostMap()
    cost_map.init_cost_map(grid_map)
    cost_map.compute_cost_map(grid_map)
    print(f'Карта стоимости создана, макс. стоимость: {cost_map.max_cost}')

    # Инициализируем MCL
    mcl = MonteCarloLocalization()
    if not initialize_particles_at_origin(mcl, cost_map, 150):
        return [], [], map_2d, []

    # Обрабатываем все доступные данные
    num_data = min(len(path_data.poses), len(laserscans))  # Вся доступная траектория
    estimated_poses = []
    true_poses = []
    weights_history = []

    prev_pose = None

    for i in range(num_data):
        print(f'Обработка измерения {i + 1}/{num_data}')

        # Получаем истинную позу
        true_pose = extract_pose_from_stamped(path_data.poses[i])

        # Предсказание по одометрии (только при значительном движении)
        if prev_pose is not None:
            delta_dist, delta_theta = compute_odometry_delta(prev_pose, true_pose)
            if delta_dist > 0.005 or abs(delta_theta) > 0.01:  # Больший порог
                mcl.estimate_from_odometry(delta_dist, delta_theta)
                print(f'  Одометрия: dist={delta_dist:.3f}, theta={delta_theta:.3f}')

        # Получаем измерения лидара
        measurements = convert_laserscan_to_measurements(laserscans[i])

        # Обновление весов частиц
        if len(measurements) > 0:
            mcl.evaluate_particles(cost_map, measurements)
            max_weight = mcl.max_weight
            avg_weight = np.mean([p.weight for p in mcl.particles])
            weights_history.append((max_weight, avg_weight))
            print(f'  Макс. вес: {max_weight:.6f}, Средний вес: {avg_weight:.6f}')

        # Консервативная передискретизация - только когда веса сильно разошлись
        effective_sample_size = 1.0 / sum(p.weight**2 for p in mcl.particles)
        print(f'  Эффективный размер выборки: {effective_sample_size:.1f}')

        # Обнаружение больших ошибок и восстановление
        if len(estimated_poses) > 0:
            last_est = estimated_poses[-1]
            current_error = sqrt((true_pose.x - last_est.x) ** 2 + (true_pose.z - last_est.z) ** 2)

            # Консервативная коррекция только при критических ошибках
            if current_error > 2.0:  # Увеличиваем порог
                print(f'  КРИТИЧЕСКАЯ ОШИБКА {current_error:.3f}м - добавляем 5% случайных частиц')
                mcl.add_random_particles(cost_map, 5)  # Уменьшаем количество
                mcl.re_sampling()

        if effective_sample_size < len(mcl.particles) * 0.25:  # Более консервативный порог
            print('  Выполняем передискретизацию...')
            mcl.re_sampling()
            # Редкое добавление случайных частиц
            if i % 20 == 0:  # Каждые 20 шагов вместо 5
                mcl.add_random_particles(cost_map, 2)  # 2% вместо 5%

        # Получаем оценку позиции
        if mcl.id_best_particle is not None:
            best_particle = mcl.particles[mcl.id_best_particle]
            estimated_poses.append(copy.copy(best_particle.pose))

            # Вычисляем ошибку
            error_x = true_pose.x - best_particle.pose.x
            error_z = true_pose.z - best_particle.pose.z
            error_dist = sqrt(error_x * error_x + error_z * error_z)
            print(f'  Истинная: ({true_pose.x:.3f}, {true_pose.z:.3f})')
            print(f'  Оценочная: ({best_particle.pose.x:.3f}, {best_particle.pose.z:.3f})')
            print(f'  Ошибка: {error_dist:.3f}м')

        true_poses.append(copy.copy(true_pose))
        prev_pose = true_pose
        print()

    print('Улучшенный AMCL завершен!')

    # Вычисляем финальную статистику
    if len(estimated_poses) > 0:
        errors = []
        for true_pose, est_pose in zip(true_poses[-len(estimated_poses) :], estimated_poses):
            error_x = true_pose.x - est_pose.x
            error_z = true_pose.z - est_pose.z
            error_dist = sqrt(error_x * error_x + error_z * error_z)
            errors.append(error_dist)

        avg_error = np.mean(errors)
        median_error = np.median(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)

        print('Статистика ошибок:')
        print(f'  Средняя: {avg_error:.3f}м')
        print(f'  Медианная: {median_error:.3f}м')
        print(f'  Максимальная: {max_error:.3f}м')
        print(f'  Минимальная: {min_error:.3f}м')

    return estimated_poses, true_poses, map_2d, weights_history


def visualize_improved_results(estimated_poses, true_poses, map_2d, weights_history):
    """Визуализирует результаты с дополнительной статистикой"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Траектории
    map_display = 1 - map_2d / 100
    ax1.imshow(map_display, cmap='gray', origin='lower', extent=[-10, 10, -10, 10], alpha=0.7)

    true_x = [pose.x for pose in true_poses]
    true_z = [pose.z for pose in true_poses]
    ax1.plot(true_x, true_z, 'g-', linewidth=3, label='Истинная траектория', alpha=0.8)
    ax1.plot(true_x[0], true_z[0], 'go', markersize=12, label='Старт')

    if estimated_poses:
        est_x = [pose.x for pose in estimated_poses]
        est_z = [pose.z for pose in estimated_poses]
        ax1.plot(est_x, est_z, 'r-', linewidth=3, label='Оцененная траектория', alpha=0.8)
        ax1.plot(est_x[0], est_z[0], 'ro', markersize=12, label='Старт (оценка)')
        ax1.plot(est_x[-1], est_z[-1], 'r^', markersize=15, label='Финал (оценка)')

    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Z (м)')
    ax1.set_title('Траектории')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # График 2: Ошибки по времени
    if estimated_poses:
        errors = []
        for true_pose, est_pose in zip(true_poses[-len(estimated_poses) :], estimated_poses):
            error_x = true_pose.x - est_pose.x
            error_z = true_pose.z - est_pose.z
            error_dist = sqrt(error_x * error_x + error_z * error_z)
            errors.append(error_dist)

        ax2.plot(errors, 'b-', linewidth=2)
        ax2.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Среднее: {np.mean(errors):.3f}м')
        ax2.set_xlabel('Номер измерения')
        ax2.set_ylabel('Ошибка позиции (м)')
        ax2.set_title('Ошибка локализации по времени')
        ax2.legend()
        ax2.grid(True)

    # График 3: Веса частиц
    if weights_history:
        max_weights = [w[0] for w in weights_history]
        avg_weights = [w[1] for w in weights_history]

        ax3.plot(max_weights, 'r-', linewidth=2, label='Максимальный вес')
        ax3.plot(avg_weights, 'b-', linewidth=2, label='Средний вес')
        ax3.set_xlabel('Номер измерения')
        ax3.set_ylabel('Вес частицы')
        ax3.set_title('Эволюция весов частиц')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')

    # График 4: Распределение ошибок
    if estimated_poses:
        ax4.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax4.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Среднее: {np.mean(errors):.3f}м')
        ax4.axvline(x=np.median(errors), color='g', linestyle='--', label=f'Медиана: {np.median(errors):.3f}м')
        ax4.set_xlabel('Ошибка позиции (м)')
        ax4.set_ylabel('Частота')
        ax4.set_title('Распределение ошибок')
        ax4.legend()
        ax4.grid(True)

    plt.tight_layout()
    plt.savefig('improved_amcl_result.png', dpi=300, bbox_inches='tight')
    print('Улучшенный результат сохранен в improved_amcl_result.png')
    plt.show()


if __name__ == '__main__':
    estimated_poses, true_poses, map_2d, weights_history = run_amcl()
    if estimated_poses:
        visualize_improved_results(estimated_poses, true_poses, map_2d, weights_history)
