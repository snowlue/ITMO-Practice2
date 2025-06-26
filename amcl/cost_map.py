"""
This file defines a CostCell class and a CostMap class
"""

from math import cos, sin

from environment import GridCell, GridMap
from robot import LiDARMeasurement, Pose3D


class CostCell(GridCell):
    """class to handle the cells of a cost map
    ATTRIBUTES:
        cost : (number) the cost of the cell
    """

    cost: float

    def __init__(self, x=0, z=0, cost=0):
        """constructor of the class
        :param x: (int) x coordinate of the cell in the map
        :param z: (int) z coordinate of the cell in the map
        :param cost: (number) the cost of the cell
        """
        GridCell.__init__(self, x, z)
        self.cost = cost  # cost of the cell (distance to the closest obstacle)

    def __str__(self) -> str:
        """to display a CostCell as a string"""
        return f'({self.x}, {self.z}, {self.cost})'


class CostMap(GridMap):
    """class to handle a cost map
    ATTRIBUTES:
        max_cost: (number) maximal current cost of the cells
    """

    max_cost: float

    def __init__(self):
        """constructor of the class"""
        GridMap.__init__(self)
        self.max_cost = None  # type: ignore # -  maximal current cost of the cells

    def init_cost_map(self, grid_map: GridMap):
        """function to initialize the cost map according to a grid map
        :param grid_map: (environment.grid_map GridMap) a GridMap to compute the cost of
        """
        self.width = grid_map.width
        self.height = grid_map.height
        self.nb_cell_x = grid_map.nb_cell_x
        self.nb_cell_z = grid_map.nb_cell_z
        self.max_cost = 0

        self.cells = [[CostCell()]]
        self.cells.clear()
        for z in range(0, self.nb_cell_z):
            self.cells.append([])
            for x in range(0, self.nb_cell_x):
                c = CostCell()
                c.x = x
                c.z = z
                # if the cell corresponds to an obstacle, its cost is 0
                # otherwise the cost is initialized by +infinity
                if grid_map.cells[z][x].val == 1:
                    c.cost = 0
                else:
                    c.cost = float('inf')
                self.cells[z].append(c)

        self.size_x = self.width / self.nb_cell_x
        self.size_z = self.height / self.nb_cell_z

    def compute_cost_map(self, grid_map: GridMap):
        """function to compute the cost map from a grid map
        :param grid_map: (environment.grid_map GridMap)
        """
        self.init_cost_map(grid_map)
        # the computation is done by going through all the cells 4 times
        # each line from the left to the right
        # each line from the right to the left
        # each row from the up to the down
        # each row from the down to the up
        # do not forget to UPDATE self.max_cost, initialized at 0

        # Выполняем итерации для распространения расстояний
        for _ in range(max(self.nb_cell_x, self.nb_cell_z)):
            self.compute_west_2_east()
            self.compute_east_2_west()
            self.compute_south_2_north()
            self.compute_north_2_south()

    def compute_west_2_east(self):
        """function to compute the cells from the west corner to the east side"""
        for z in range(self.nb_cell_z):
            for x in range(1, self.nb_cell_x):
                if self.cells[z][x].cost != 0:  # Не обрабатываем препятствия
                    new_cost = self.cells[z][x - 1].cost + 1
                    if new_cost < self.cells[z][x].cost:
                        self.cells[z][x].cost = new_cost
                        if new_cost > self.max_cost:
                            self.max_cost = new_cost

    def compute_east_2_west(self):
        """function to compute the cells from the east to the west side"""
        for z in range(self.nb_cell_z):
            for x in range(self.nb_cell_x - 2, -1, -1):
                if self.cells[z][x].cost != 0:  # Не обрабатываем препятствия
                    new_cost = self.cells[z][x + 1].cost + 1
                    if new_cost < self.cells[z][x].cost:
                        self.cells[z][x].cost = new_cost
                        if new_cost > self.max_cost:
                            self.max_cost = new_cost

    def compute_south_2_north(self):
        """function to compute the cells the top to the bottom"""
        for x in range(self.nb_cell_x):
            for z in range(1, self.nb_cell_z):
                if self.cells[z][x].cost != 0:  # Не обрабатываем препятствия
                    new_cost = self.cells[z - 1][x].cost + 1
                    if new_cost < self.cells[z][x].cost:
                        self.cells[z][x].cost = new_cost
                        if new_cost > self.max_cost:
                            self.max_cost = new_cost

    def compute_north_2_south(self):
        """function to compute the cells from the bottom to the top"""
        for x in range(self.nb_cell_x):
            for z in range(self.nb_cell_z - 2, -1, -1):
                if self.cells[z][x].cost != 0:  # Не обрабатываем препятствия
                    new_cost = self.cells[z + 1][x].cost + 1
                    if new_cost < self.cells[z][x].cost:
                        self.cells[z][x].cost = new_cost
                        if new_cost > self.max_cost:
                            self.max_cost = new_cost

    def cell_is_empty(self, dx: float, dz: float) -> bool:
        """function to test if the cell corresponding to the dx and dz distances is empty or not
        dx and dz are coordinates in the environment
        PARAMETERS:
            dx: (number in m) x coordinate in the world frame
            dz: (number in m) z coordinate in the world frame
        RETURNS:
            True of False
        """
        # Преобразуем мировые координаты (-10, 10) в индексы сетки (0, 400)
        # dx, dz могут быть от -10 до +10, нужно преобразовать в 0-400
        x_idx = int((dx + 10.0) / self.size_x)
        z_idx = int((dz + 10.0) / self.size_z)

        # Проверяем границы
        if x_idx < 0 or x_idx >= self.nb_cell_x or z_idx < 0 or z_idx >= self.nb_cell_z:
            return False

        # Проверяем, что ячейка свободна
        # В cost map: препятствие имеет cost = 0, свободная область cost > 0
        # Мы добавляем частицы только в области с умеренной стоимостью
        # (не в препятствия и не слишком далеко от них)
        cost = self.cells[z_idx][x_idx].cost

        # Свободная ячейка должна иметь cost > 0 (не препятствие)
        # и cost != inf (достижимая область)
        # и cost < max_cost * 0.8 (не слишком далеко от препятствий)
        if cost > 0 and cost != float('inf') and cost < self.max_cost * 0.8:
            return True
        else:
            return False

    def evaluate_cost(self, pose: Pose3D, measurements: list[LiDARMeasurement]) -> float:
        """function to evaluate the cost of measurements set according to the
        position x,z and the orientation theta
        parameters:
            pose: (robot.pose.Pose3D) the pose (x, z, theta)
            measurements: (list of robot.lidar.LiDARMeasurement) the measurements
        return:
            cost: (float)
        """
        cost = 0.0
        valid_measurements = 0

        for measurement in measurements:
            # Вычисляем ожидаемую позицию конца луча лидара
            expected_x = pose.x + measurement.distance * cos(pose.theta + measurement.angle)
            expected_z = pose.z + measurement.distance * sin(pose.theta + measurement.angle)

            # Преобразуем мировые координаты (-10, 10) в индексы сетки (0, 400)
            x_idx = int((expected_x + 10.0) / self.size_x)
            z_idx = int((expected_z + 10.0) / self.size_z)

            # Проверяем границы
            if x_idx < 0 or x_idx >= self.nb_cell_x or z_idx < 0 or z_idx >= self.nb_cell_z:
                cost += 50  # Штраф за выход за границы
                continue

            valid_measurements += 1

            # Инвертируем логику: если луч заканчивается рядом с препятствием, это хорошо
            cell_cost = self.cells[z_idx][x_idx].cost
            if cell_cost == 0:
                # Луч попал в препятствие - отлично!
                cost += 0
            elif cell_cost == float('inf'):
                # Луч попал в недостижимую область
                cost += 100
            else:
                # Луч попал в свободную область, но близко к препятствию - тоже хорошо
                # Меньшая стоимость (ближе к препятствию) = меньший штраф
                cost += min(cell_cost, 50)  # Ограничиваем максимальный штраф

        # Нормализуем по количеству валидных измерений
        if valid_measurements > 0:
            cost = cost / valid_measurements
        else:
            cost = 1000  # Большой штраф если нет валидных измерений

        return cost
