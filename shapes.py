from __future__ import annotations
from collections.abc import Sequence
import numpy as np

# https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
"""class Point:
  def __init__(self, x: float, y: float):
    self.x = x
    self.y = y
    self.pt = np.array([x, y])
"""


def arr_eq(a: np.ndarray, b: np.ndarray) -> bool:
    """Tests if two arrays are equal element-wise"""
    return np.all(np.equal(a, b))


Point = np.ndarray


def point(x: float | np.ndarray, y: float | np.ndarray):
    return np.array([x, y])


class Edge:
    def __init__(self, p1: Point, p2: Point):
        # Do the checks later
        self.pt_a = p1
        self.pt_b = p2
        try:
            test = np.equal(self.pt_a, self.pt_b)
            assert not np.all(test)
        except AssertionError:
            raise ValueError("Cannot create an edge from two identical points.")

        # Set slope
        x1, y1 = self.pt_a
        x2, y2 = self.pt_b
        del_x = x2 - x1
        if del_x == 0:
            self.slope = np.inf
            self.b = None
        else:
            self.slope = (y2 - y1)/del_x
            self.b = (-self.slope * x1) + y1

    def __repr__(self):
        string = f'({self.pt_a[0]}, {self.pt_a[1]}) -> ({self.pt_b[0]}, {self.pt_b[1]})'
        return string

    def is_horiz(self):
        y1, y2 = self.pt_a[1], self.pt_b[1]
        return y1 == y2

    def is_vert(self):
        x1, x2 = self.pt_a[0], self.pt_b[0]
        return x1 == x2

    def get_y_eqn(self):
        if self.is_horiz():
            def eqn(x):
                return self.pt_a[1]
        elif self.is_vert():
            reroute = self.get_x_eqn()
            return reroute
        else:
            def eqn(x):
                x1, y1 = self.pt_a
                x2, y2 = self.pt_b
                m = (y2 - y1) / (x2 - x1)
                return m * (x - x1) + y1
        return eqn

    def get_x_eqn(self):
        if self.is_horiz():
            reroute = self.get_y_eqn()
            return reroute
        elif self.is_vert():
            def eqn(y):
                return self.pt_a[0]
        else:
            def eqn(y):
                x1, y1 = self.pt_a
                x2, y2 = self.pt_b
                m = (y2 - y1) / (x2 - x1)
                return (y - y1) / m + x1
        return eqn

    def which_y_dir(self, pt: Point) -> int:
        """
        Determine if a given point is above the edge, on it, or below it

        Outputs:
            1: pt is above edge
            0: pt is on edge
            -1: pt is below edge
        """
        result = 0
        x, y = pt[:2]
        eqn = self.get_y_eqn()
        test_y = eqn(x)
        if y > test_y:
            result += 1
        elif y < test_y:
            result -= 1
        return result

    def which_x_dir(self, pt: Point) -> int:
        """
        Determine if a given point is to the left of the edge, on it, or right of it

        Outputs:
            1: pt is right of edge
            0: pt is on edge
            -1: pt is left of edge
        """
        result = 0
        x, y = pt[:2]
        eqn = self.get_x_eqn()
        test_x = eqn(y)
        if x > test_x:
            result += 1
        elif x < test_x:
            result -= 1
        return result

    def is_on(self, pt: Point) -> bool:
        # print(f'<is_on> Edge: {self}')
        # print(f'<is_on> Point: {pt}')
        out = False
        x1, y1 = self.pt_a
        x2, y2 = self.pt_b
        pt_x, pt_y = pt
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        if (
                (self.which_y_dir(pt) == 0 or self.which_x_dir(pt) == 0) and
                (x_min <= pt_x <= x_max and y_min <= pt_y <= y_max)
        ):
            out = True
        # print(f'<is_on> {out}')
        return out

    def is_parallel(self, edge2: 'Edge'):
        m1 = self.slope
        m2 = edge2.slope
        if (np.isinf(m1) and np.isinf(m2)) or m1 == m2:
            return True
        else:
            return False

    def find_intersection(self, k: 'Edge', independent_var: str = 'x'):
        """Returns a function that calculates the intersection point"""
        # print(f'Edge 1: {self}')
        # print(f'Edge 2: {k}')
        output = ()
        if self.is_parallel(k):
            output = None
        elif self.slope is np.inf:
            x = self.pt_a[0]
            y_val = k.get_y_eqn()(x)
            output = point(x, y_val)
        elif k.slope is np.inf:
            x = k.pt_a[0]
            y_val = self.get_y_eqn()(x)
            output = point(x, y_val)
        else:
            A = np.array([
                [-self.slope, 1],
                [-k.slope, 1]
            ])
            b = np.array([
                [self.b],
                [k.b]
            ])
            # print(A)
            # print(b)
            intr = np.linalg.inv(A) @ b
            output = np.transpose(intr).flatten()
        # print(f'Intersection: {output}\n')
        return output


class Polygon:
    def __init__(self, connectors: Sequence, sides: int):
        """
        connectors values must be ordered to have either CCW or CW point progression
        """
        if len(connectors) != sides:
            raise ValueError("Must have four entries define the quadrilateral.")
        self.num_sides = sides
        if all([isinstance(x, Point) for x in connectors]):
            # define edges
            self.edges = [Edge(connectors[i - 1], connectors[i]) for i in range(1, sides)]
            self.edges.append(Edge(connectors[-1], connectors[0]))  # close the shape
        elif all([isinstance(x, Edge) for x in connectors]):
            first, last = connectors[0], connectors[-1]
            if not arr_eq(first.pt_a, last.pt_b):
                raise ValueError(
                    ("Edges do not close into shape. Ensure entries proceed CW"
                     " starting from the left-most edge.")
                )
            try:
                assert all([arr_eq(connectors[i - 1].pt_b, connectors[i].pt_a) for i in range(1, sides)])
            except AssertionError:
                print('Ensure edges form a closed shape.')
                raise
            self.edges = connectors
        else:
            raise TypeError(
                (f"connectors must all be of type {type(Point)} "
                 f"or must all be of type {type(Edge)}")
            )
        self.vertices = [x.pt_a for x in self.edges]

    def __repr__(self):
        cat = "".join
        strings = [cat([edge.__repr__(), '\n']) for edge in self.edges]
        strings.insert(0, '\n')
        string = cat(strings)
        return string

    def contains(self, pt: Point) -> bool:
        """Determines if the provided point is lies
        within the bounds of the polygon."""

        # https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
        result = False
        mag = np.linalg.norm
        tolerance = 1e-10
        target = 2 * np.pi
        vertices = self.vertices
        angle = 0
        if any([arr_eq(pt, x) for x in vertices]):
            return False
        # print(f'Point: {pt}')
        for i in range(self.num_sides):
            vertex1 = vertices[i]
            vertex2 = vertices[(i + 1) % self.num_sides]
            # print(f"Points: {vertex1}\n{vertex2}")
            # Create vectors
            v1 = vertex1 - pt
            v2 = vertex2 - pt
            temp_angle = np.arccos(np.dot(v1, v2)/(mag(v1) * mag(v2)))
            # print("Vec1, Vec2, calc angle:")
            # print(v1, v2, temp_angle)
            angle += temp_angle
            # print(angle)
        else:
            if abs(target - angle) <= tolerance:
                result = not result
        return result

    def touches(self, pt: Point) -> bool:
        """Does the provided point lie on an edge?"""
        for edge in self.edges:
            if edge.is_on(pt):
                return True
        else:
            return False

    def on_or_in(self, pt: Point) -> bool:
        return self.contains(pt) or self.touches(pt)


class Quadrilateral(Polygon):
    def __init__(self, connectors: Sequence):
        super().__init__(connectors, 4)
        # Define bounds
        self.left, self.top, self.right, self.bottom = self.edges

    def contains(self, pt: Point) -> bool:
        result = super().contains(pt)
        """
        A point is contained within the quadrilateral if it is all of the following:
        - to the right of the left edge (+1)
        - below the top edge (-1)
        - left of the right edge (-1)
        - above the bottom edge (+1)
        """
        """# Retrieve equations
        check = 0
        for k in range(self.num_sides):
            edge = self.edges[k]
            if k % 2 == 0:
                # check left and right
                adjust = edge.which_x_dir(pt)
            else:
                adjust = edge.which_y_dir(pt)
            check += adjust
            # print(f"Container Check: {check}\nAdjust: {adjust}")
        return check == 0
    """
        return result

    def touches(self, pt: Point) -> bool:
        for edge in self.edges:
            if edge.is_on(pt):
                return True
        else:
            return False

    '''
    def on_or_in(self, pt: Point) -> bool:
        """Polygon.contains does this via the winding number approach"""
        return self.contains(pt) or self.touches(pt)
    '''


class Triangle(Polygon):
    def __init__(self, connectors: Sequence):
        super().__init__(connectors, sides=3)


class Pentagon(Polygon):
    def __init__(self, connectors: Sequence):
        super().__init__(connectors, sides=5)
