"""
This file provides an SegEnv class, an environment based on a list of segments and a GridMap class and a GridCell class
"""

from geometry import Point2D, Segment2D


class SegEnv:
    """SegEnv class to handle an environment composed of a set of segments
    ATTRIBUTES:
        width: (number in m) the width of the environment
        height: (number in m) the height of the environment
        segments: (list of geometry.segment Segment2D) the segments of the environment
    METHODS:
        add(segment2d)
        init_environment()
        init_environment()

    """

    width: float
    height: float
    segments: list[Segment2D]

    def __init__(self, width: float = 0, height: float = 0):
        """the constructor of the class
        in this version the environment is a set of segments
        the list of segments is initialized as empty
        PARAMETERS:
            width: (number in m) the width of the environment
            height: (number in m) the height of the environment
        """
        self.width = width
        self.height = height
        self.segments = []

    def add(self, segment2d: Segment2D):
        """a function to add a segment to the environment
        PARAMETERS:
            segment2d: (geometry.segment Segment2D) the segment you want to add
        """
        offset = 1
        segment2d.p1.x += offset
        segment2d.p2.x += offset
        segment2d.p1.z += offset
        segment2d.p2.z += offset
        self.segments.append(segment2d)

    def init_environment(self):
        """a function to initialize the environment"""
        self.add(Segment2D(Point2D(0, 0), Point2D(10, 0)))
        self.add(Segment2D(Point2D(10, 0), Point2D(10, 10)))
        self.add(Segment2D(Point2D(10, 10), Point2D(0, 10)))
        self.add(Segment2D(Point2D(0, 10), Point2D(0, 0)))

        self.add(Segment2D(Point2D(0, 2), Point2D(5, 2)))
        self.add(Segment2D(Point2D(0, 4), Point2D(3, 4)))
        self.add(Segment2D(Point2D(0, 6), Point2D(3, 6)))
        self.add(Segment2D(Point2D(0, 8), Point2D(2, 8)))
        self.add(Segment2D(Point2D(4, 8), Point2D(10, 8)))
        self.add(Segment2D(Point2D(7, 2), Point2D(10, 2)))
        self.add(Segment2D(Point2D(7, 2), Point2D(7, 6)))
        self.add(Segment2D(Point2D(7, 6), Point2D(8, 6)))

        self.width = 12.0
        self.height = 12.0

    def init_environment_icp(self):
        """a function to initialize the environment"""
        self.add(Segment2D(Point2D(2, 6), Point2D(2, 2)))
        self.add(Segment2D(Point2D(2, 2), Point2D(6, 2)))
        self.add(Segment2D(Point2D(6, 2), Point2D(6, 6)))

        self.width = 10.0
        self.height = 10.0


class GridCell:
    """class to handle a cell in the grid map"""

    x: int  # (int) x coordinate of the cell in the map
    z: int  # (int) z coordinate of the cell in the map
    val: float  # (float) value of the cell (obstacle or free)

    def __init__(self, x: int = 0, z: int = 0, val: float = 0):
        """constructor of the class
        PARAMETERS:
            x: (int) x coordinate of the cell in the map
            z: (int) z coordinate of the cell in the map
            val: (number) value of the cell (obstacle or free)
        """
        self.x = x  # x coordinate of the cell in the map
        self.z = z  # y coordinate of the cell in the map
        self.val = val  # value of the cell (obstacle or free)


class GridMap:
    """class to handle a grid map

    METHODS:
        init_map()
        def compute_map(environment)
    """

    width: float  # (number in m)  the width of the map
    height: float  # (number in m) the height of the map
    nb_cell_x: int  # (int) number of cells according to the x-axis
    nb_cell_z: int  # (int) number of cells according to the z-axis
    cells: list[list[GridCell]]  # (list of GridCell)  list of cells
    size_x: float  # (number in m) x size of a cell
    size_z: float  # (number in m) z size of a cell

    def __init__(self):
        """constructor of the class"""
        self.width = None  # type: ignore # m  the width of the map
        self.height = None  # type: ignore # m  the height of the map
        self.nb_cell_x = None  # type: ignore # -  number of cells according to the x-axis
        self.nb_cell_z = None  # type: ignore # -  number of cells according to the z-axis
        self.cells = None  # type: ignore # -  list of cells
        self.size_x = None  # type: ignore # m  x size of a cell
        self.size_z = None  # type: ignore # m  z size of a cell

    def init_map(self):
        """function to initialize the map"""
        self.width = 12.0
        self.height = 12.0
        self.nb_cell_x = 60
        self.nb_cell_z = 60

        self.cells = []
        for z in range(0, self.nb_cell_z):
            self.cells.append([])
            for x in range(0, self.nb_cell_x):
                c = GridCell()
                c.x = x
                c.z = z
                c.val = 0.0
                self.cells[z].append(c)

        self.size_x = self.width / self.nb_cell_x
        self.size_z = self.height / self.nb_cell_z

    def compute_map(self, environment: SegEnv):
        """function to compute the map according to the environment
        if the cell intersect a segment of the environment it is an obstacle (1)
        PARAMETERS:
            environment: (environment.seg_environment SegEnv) the environment we want to compute the grid map of
        """
        for line in self.cells:
            for cell in line:
                cell.val = 0.0
                for segment in environment.segments:
                    a = Point2D(cell.x * self.size_x, cell.z * self.size_z)
                    b = Point2D((cell.x + 1) * self.size_x, cell.z * self.size_z)
                    c = Point2D((cell.x + 1) * self.size_x, (cell.z + 1) * self.size_z)
                    d = Point2D(cell.x * self.size_x, (cell.z + 1) * self.size_z)

                    seg_ab = Segment2D(a, b)
                    seg_bc = Segment2D(b, c)
                    seg_cd = Segment2D(c, d)
                    seg_da = Segment2D(d, a)

                    inter_ab = Segment2D.intersect(seg_ab, segment)
                    inter_bc = Segment2D.intersect(seg_bc, segment)
                    inter_cd = Segment2D.intersect(seg_cd, segment)
                    inter_da = Segment2D.intersect(seg_da, segment)

                    if inter_ab[0] or inter_bc[0] or inter_cd[0] or inter_da[0]:
                        cell.val = 1.0

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            li, col = key
            self.cells[li][col] = value
        else:
            self.cells[key] = value

    def __getitem__(self, key):
        if isinstance(key, tuple):
            li, col = key
            return self.cells[li][col]
        else:
            return self.cells[key]

