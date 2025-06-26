"""
This file provides a Point2D class to manipulate 2D points and a Segment2D class to manipulate 2D segments
"""

from math import sqrt


class Point2D:
    """class to handle 2 dimensional points
    ATTRIBUTES:
        self.x : (number) the x coordinate of the point
        self.z : (number) the z coordinate of the point
    """

    x: float
    z: float

    def __init__(self, x: float = 0, z: float = 0):
        """constructor of the class
        PARAMETERS:
            x : (number) the x coordinate of the point
            z : (number) the z coordinate of the point
        """
        self.x = x
        self.z = z

    def __str__(self) -> str:
        """to be able to print a Point2D"""
        return '(' + str(self.x) + ',' + str(self.z) + ')'

    def __eq__(self, other) -> bool:
        """to test the equality between to Point2D
        PARAMETERS:
            other : (geometry.point.Point2D) the point to compare self with
        """
        return other.x == self.x and other.z == self.z

    def __repr__(self) -> str:
        """function that returns a printable representation of the self Point2D object"""
        return f'{self}'

    @staticmethod
    def distance(p1: 'Point2D', p2: 'Point2D') -> float:
        """Function that computes the euclidean distance between two points
        PARAMETERS:
            p1 : (Point2D) the first point
            p2 : (Point2D) the second point
        RETURNS:
            the distance (number)
        """
        return sqrt((p1.x - p2.x) ** 2 + (p1.z - p2.z) ** 2)


class Segment2D:
    """class to manipulate 2 dimensional segment
    ATTRIBUTES:
        self.p1 : (geometry.point Point2D) first point of the segment
        self.p2 : (geometry.point Point2D) second point of the segment
    """

    p1: Point2D
    p2: Point2D

    def __init__(self, p1: Point2D = None, p2: Point2D = None):  # type: ignore
        """constructor of the class
        PARAMETERS:
            p1 : (geometry.point Point2D) first point of the segment
            p2 : (geometry.point Point2D) second point of the segment
        """
        if p1 is None:
            p1 = Point2D()
        if p2 is None:
            p2 = Point2D()
        self.p1 = p1
        self.p2 = p2

    def __str__(self) -> str:
        """to be able to print a Segment2D"""
        return '[ ' + str(self.p1) + ' - ' + str(self.p2) + ' ]'

    @staticmethod
    def intersect(s1: 'Segment2D', s2: 'Segment2D') -> tuple[bool, Point2D]:
        """static function to test the intersection of two Segment2D
        PARAMETERS:
            s1 : (Segment2D) first segment
            s2 : (Segment2D) second segment
        RETURNS:
            True or False
            the intersection point (Point2D)
        """
        # using intermediate variables to ease the reading
        ax = s1.p1.x
        az = s1.p1.z
        bx = s1.p2.x
        bz = s1.p2.z
        cx = s2.p1.x
        cz = s2.p1.z
        dx = s2.p2.x
        dz = s2.p2.z

        ux = bx - ax
        uz = bz - az
        vx = dx - cx
        vz = dz - cz

        denominator = vz * ux - vx * uz

        if denominator == 0:
            # the lines (defined by the segments) are parallel
            if Segment2D.belongs(s1.p1, s2):
                return True, s1.p1
            elif Segment2D.belongs(s1.p2, s2):
                return True, s1.p2
            elif Segment2D.belongs(s2.p1, s1):
                return True, s2.p1
            elif Segment2D.belongs(s2.p2, s1):
                return True, s2.p2
            else:
                return False, Point2D(0, 0)
        else:
            # the lines (defined by the segments) are not parallel
            # we use the parametric equation of the lines
            q = ((cx - ax) * uz - (cz - az) * ux) / (denominator * 1.0)
            t = ((ax - cx) * vz - (az - cz) * vx) / (-denominator * 1.0)
            if 0 < q < 1 and 0 < t < 1:
                # the segments do intersect
                # we compute the intersection point
                pinter = Point2D(ax + t * ux, az + t * uz)
                return True, pinter
            else:
                # the lines do intersect but not the segments
                return False, Point2D(0, 0)

    @staticmethod
    def det(a: Point2D, c: Point2D, b: Point2D) -> float:
        """static function to compute the determinant
        PARAMETERS:
            a : (geometry.point Point2D)
            c : (geometry.point Point2D)
            b : (geometry.point Point2D)
        RETURNS:
            the determinant (number)
        """
        return (a.x - b.x) * (c.z - b.z) - (a.z - b.z) * (c.x - b.x)

    @staticmethod
    def does_intersect(s1: 'Segment2D', s2: 'Segment2D') -> bool:
        """static function that test if two segment intersect without computing the intersection point
        PARAMETERS:
            s1 : (Segment2D) first segment
            s2 : (Segment2D) second segment
        RETURNS:
            True or False
        """
        # variables to ease the reading
        a = s1.p1
        b = s1.p2
        c = s2.p1
        d = s2.p2

        test1 = Segment2D.det(a, c, d) * Segment2D.det(b, c, d)
        test2 = Segment2D.det(c, a, b) * Segment2D.det(d, a, b)

        if test1 > 0 or test2 > 0:
            return False
        else:
            return True

    @staticmethod
    def belongs(c: Point2D, seg: 'Segment2D') -> bool:
        """static function that test if a point2D C belongs to a segment2D seg
        PARAMETERS:
            c : (geometry.point Point2D) a point
            seg : (Segment2D) a segment
        RETURNS:
            True or False
        """
        a = seg.p1
        b = seg.p2

        acx = c.x - a.x
        acz = c.z - a.z

        abx = b.x - a.x
        abz = b.z - a.z

        product_vector_ab_ac = abx * acz - abz * acx

        if product_vector_ab_ac != 0:
            return False
        else:
            k_ac = abx * acx + abz * acz
            k_ab = abx * abx + abz * abz

            if k_ac < 0 or k_ac > k_ab:
                return False
            else:
                return True
