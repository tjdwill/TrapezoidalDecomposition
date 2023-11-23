from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from shapes import (
    Edge, Triangle, Pentagon, point, arr_eq, Polygon,  # Quadrilateral
)

MIN_X, MAX_X = 0, 50
MIN_Y, MAX_Y = MIN_X, MAX_X


def get_plot_data(edges: list) -> tuple:
    """Convert points to plot-compatible format"""
    x = [(line.pt_a[0], line.pt_b[0]) for line in edges]
    y = [(line.pt_a[1], line.pt_b[1]) for line in edges]

    return x, y


def flatten_data(data):
    flat = [m for item in data for m in item]
    return flat


"""
    Quadrilateral([
        Edge(point(10, 27), point(20, 30)).find_intersection(
            Edge(point(3, 31), point(8, 17))
        ),
        point(3, 31),
        point(17, 37),
        point(20, 30)
    ]),
    Quadrilateral([
        point(8, 17),
        Edge(point(10, 27), point(20, 30)).find_intersection(
            Edge(point(3, 31), point(8, 17))
        ),
        point(10, 27),
        point(12, 20)
    ]),
"""

obstacles = [
    Triangle([
        point(36, 26),
        point(48, 37),
        point(48, 21)
    ]),
    Pentagon([
        point(16, 14),
        point(23, 22),
        point(34, 21),
        point(33, 7),
        point(24, 4)
    ]),
    Polygon(
        connectors=[
            point(8, 17),
            point(3, 31),
            point(17, 37),
            point(20, 30),
            point(10, 27),
            point(12, 20)
        ],
        sides=6
    ),
]

# vertices = [tuple(y) for x in obstacles for y in x.vertices]  # np.ndarrays are non-hashable, so convert to tuple
# vertices = {*vertices}
# print(vertices)
# print('')

vertical_segments: List[Edge] = []
patches = []
for obst in obstacles:
    # print(f'Obstacle Tested: {obst}')

    filtered = [x for x in obstacles if x is not obst]
    temp_patch = []
    for vertex in obst.vertices:
        # print(f'Vertex: {vertex}')
        # print(f"Current Vertical Segments: {vertical_segments}")
        """
        Create two vertical edges:
            1. vertex -> top border
            2. vertex -> bottom border
        """
        top_vert = Edge(vertex, point(vertex[0], MAX_Y))
        bottom_vert = Edge(vertex, point(vertex[0], MIN_Y))
        verts = (top_vert, bottom_vert)
        for i in range(len(verts)):
            seg = verts[i]
            ...
            # print(f'Using Seg: {seg}')
            # Check to see if the edge intersects with any of the intra-shape edges
            self_intersects = False
            for edge in obst.edges:
                intr = seg.find_intersection(edge)
                # print(f"Intersection Between {seg} and {edge}: {intr}")
                """
                If the calculated intersection point is:
                    - None (the two edges are parallel)
                    - A vertex on the shape (the vertex should not count itself as an intersection)
                    - Located on or within the shape (unreachable)
                then there is a self-intersection; we can't draw the vertical segmentation line in the given
                direction.
                """
                if intr is None or arr_eq(intr, vertex):
                    continue
                elif (i == 0) and (obst.touches(intr) and intr[1] > vertex[1]):
                    self_intersects = True
                elif (i == 1) and (obst.touches(intr) and intr[1] < vertex[1]):
                    self_intersects = True
            else:
                if self_intersects:
                    continue
                else:
                    # check the other obstacles.
                    # print("Checking other obstacles...")
                    """
                    For a given obstacle edge, calculate the intersection with the vertical segment.
                    If this intersection exists and lies on the edge (not on the extended line made by the edge), then this
                    intersection point is the new landing point for the vertical segment.
                    
                    If by some chance there are multiple obstacles that intersect with the vert. seg, take the one with the
                    smallest y value.
                    
                    If there is no such point for all other obstacles, then the original vert. seg. stored in the segment
                    container.
                    """
                    intersection_ys = []
                    intr_found = False
                    for other in filtered:
                        for edge in other.edges:
                            intr = seg.find_intersection(edge)
                            # print(f'Intersection between {seg} and {edge}: {intr}')
                            if (
                                    (intr is None) or
                                    (not edge.is_on(intr)) or
                                    arr_eq(intr, vertex) or
                                    (i == 0 and intr[1] < vertex[1]) or
                                    (i == 1 and intr[1] > vertex[1]) or
                                    other.contains(intr)
                            ):
                                continue
                            else:
                                intersection_ys.append(intr[1])
                                intr_found = True
                                # print(f'Found {intr} from {seg} to {edge}')
                    else:
                        if not intr_found:
                            # Append the original vertical segment as no shorter segments were found
                            vertical_segments.append(seg)
                        else:
                            if i == 0:
                                intr_y = min(intersection_ys)
                            else:
                                intr_y = max(intersection_ys)
                            vertical_segments.append(Edge(vertex, point(vertex[0], intr_y)))
# print(len(vertical_segments))
print("Derived segments:")
for k in vertical_segments:
    print(k)


# Plot the segments and boundaries
fig, ax = plt.subplots()
ax.grid()
x_data, y_data = get_plot_data(vertical_segments)

# Make obstacle polygons
for shape in obstacles:
    verts = shape.vertices
    x, y = zip(*verts)
    data = np.transpose(np.array([x, y]))
    x = data[:, 0]
    y = data[:, 1]
    # patch = Polygon(data, closed=True, fill=True, color='g')
    # ax.add_patch(patch)
    ax.fill(x, y, 'b')
    plt.draw()

# Add segments
for j, k in zip(x_data, y_data):
    ax.plot(j, k, color='k', linestyle='--', zorder=1)

ax.set(xlabel='x', ylabel='y', title='Trapezoidal Decomposition')
ax.set_axisbelow(True)
plt.show()
