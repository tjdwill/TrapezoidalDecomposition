from shapes import *

if __name__ == '__main__':
    pt1 = point(-1, -1)
    pt2 = point(-1, 1)
    pt3 = point(1, 1)
    pt4 = point(1, -1)

    horiz = Edge(pt1, pt2)
    try:
        c2 = Edge(pt1, pt1)
    except ValueError:
        # print("Caught same point.")
        slant = Edge(pt1, pt3)

    vert = Edge(pt1, pt4)

    # Try a square
    pt1 = point(-1, -1)
    pt2 = point(-1, 1)
    pt3 = point(1, 1)
    pt4 = point(1, -1)
    pts = [pt1, pt2, pt3, pt4]
    square = Quadrilateral(pts)
    assert square.contains(point(0, 0))

    # Triangle
    pt1 = point(-1, -1)
    pt2 = point(-1, 1)
    pt3 = point(1, 0)
    pts = [pt1, pt2, pt3]
    triangle = Triangle(pts)
    # for edge in triangle.edges:
    #    print(edge)
    print(triangle.contains(point(2, 0)))  # False
    assert triangle.contains(point(0, 0))  # True

    # Pentagon
    pt1 = point(24, 4)
    pt2 = point(16, 14)
    pt3 = point(23, 22)
    pt4 = point(34, 21)
    pt5 = point(33, 7)
    pts = [pt1, pt2, pt3, pt4, pt5]
    penta = Pentagon(pts)
    print(penta)
    for pnt in pts:
        assert penta.touches(pnt)
    assert penta.contains(point(23, 14))
    assert not penta.contains(point(0, 0))

# Intersections

eg1 = Edge(
    point(0, 0),
    point(1, 1)
)
eg2 = Edge(
    point(-1, 1),
    point(2, 1)
)

print(eg1.is_parallel(eg1))  # True
print(eg1.is_parallel(eg2))  # False
intr = eg1.find_intersection(eg2)
print(intr)  # (1, 1)
print(intr.shape)
assert arr_eq(intr, point(1, 1))

eg1 = Edge(
    point(0, 0),
    point(1, 1)
)
eg2 = Edge(
    point(-1, 1),
    point(2, -2)
)
intr = eg1.find_intersection(eg2)
assert arr_eq(intr, point(0, 0))
print(intr)  # (0, 0)


