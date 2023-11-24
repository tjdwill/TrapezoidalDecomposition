"""
@title: trapezoidal_decomposition.py
@author: Terrance Williams
@date: 22 November 2023
@description: 
    This document uses definitions from `shapes.py` to perform trapezoidal decomposition on the repreesentation of a map 
    of obstacles.
"""
from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from shapes import (
    Edge, Polygon, Triangle, Pentagon, point, arr_eq
)


def get_plot_data(edges: list) -> tuple:
    """Convert points to plot-compatible format"""
    x = [(line.pt_a[0], line.pt_b[0]) for line in edges]
    y = [(line.pt_a[1], line.pt_b[1]) for line in edges]

    return x, y


def flatten_data(data):
    flat = [m for item in data for m in item]
    return flat


if __name__ == "__main__":
    # Define Border Bounds; Assume Border is a Square.
    MIN_X, MAX_X = 0, 50
    MIN_Y, MAX_Y = MIN_X, MAX_X

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
        Polygon([
            point(8, 17),
            point(3, 31),
            point(17, 37),
            point(20, 30),
            point(10, 27),
            point(12, 20)
        ]),
    ]
    vertical_segments: List[Edge] = []  # Cell Delimiters

    # Find valid vertical delimeters for each vertex in each obstacle.
    for obst in obstacles:
        # print(f'Obstacle Tested: {obst}')

        filtered = [x for x in obstacles if x is not obst]
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
                    ignore the intersection.
                    Otherwise, if
                        - Located on the shape and has a y-value exceeding the vertex in a given direction
                        (based on which vertical segment is used)
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
                        Here are the criteria for a valid intersection between vert. seg and external shape edge:
                            0. The edges are not parallel
                            1. The intersection actually lies on the shape's edge rather than
                            simply the line describing the edge.
                            2. The intersection is not identical to the current vertex.
                            3. The intersection y-value is valid for the given vertical segment.
                            4. The intersection is not contained within the shape.
                        Meeting these conditions result in the intersection point's y-value being added to the list of 
                        potential connecting points. 
                        
                        If by some chance there are multiple obstacles that intersect with the vert. seg,
                        take the one with the furthest distance to a given border. This is done to prevent the case
                        in which the resulting segment passes through another obstacle.
                        
                        If there is no valid intersection for any of the other obstacles, then the original vert. seg.
                        is stored in the segment container.
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
        ax.fill(x, y, 'b')
        plt.draw()

    # Add segments and customize plot
    for j, k in zip(x_data, y_data):
        ax.plot(j, k, color='k', linestyle='--', zorder=1)

    ax.set(xlabel='x', ylabel='y', title='Trapezoidal Decomposition')
    ax.set_axisbelow(True)
    plt.show()
