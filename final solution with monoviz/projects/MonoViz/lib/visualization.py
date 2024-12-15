#!/usr/bin/env python
import cv2
import math
import numpy as np
import random
import torch
import torch.nn.functional as F

def draw_image(image, corners, x3d, z3d, color=(0, 0, 255), thickness=2):
    face_idx = [[0,1,5,4],
                [1,2,6,5],
                [2,3,7,6],
                [3,0,4,7]]
    for ind_f in range(4):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (int(corners[f[j]][0]), int(corners[f[j]][1])),
                     (int(corners[f[(j + 1) % 4]][0]), int(corners[f[(j + 1) % 4]][1])),
                     color, thickness)
    
    # Label the specified corner with x3d and z3d
    distance = math.sqrt(x3d*x3d + z3d*z3d)
    text = f"({distance:.2f} m)"
    corner_coords = corners[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 255, 0)
    bg_color = (0, 0, 0)  # Background color (black)
    text_pos = (int(corner_coords[0]), int(corner_coords[1]))

    # Get text size
    (text_width, text_height), base_line = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Coordinates for the background rectangle
    # We'll make the rectangle slightly larger than the text itself
    rect_start = (text_pos[0], text_pos[1] + base_line)
    rect_end = (text_pos[0] + text_width, text_pos[1] - text_height - base_line)

    # Draw the background rectangle
    cv2.rectangle(image, rect_start, rect_end, bg_color, thickness=-1)

    # Put the text on top of the background rectangle
    cv2.putText(image,
                text,
                text_pos,
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA)



def rotate_point(x, y, center_x, center_y, theta):
    x = x - center_x
    y = y - center_y
    nx = int(center_x + x * math.cos(theta) - y * math.sin(theta))
    ny = int(center_y + x * math.sin(theta) + y * math.cos(theta))
    return nx, ny    

def draw_bev_rect(image, rect, x3d, z3d, thickness=2):
    # rect = [center_x, center_y, w, h, theta]
    center_x = rect[0]
    center_y = rect[1]
    w = rect[2]
    h = rect[3]
    theta = rect[4]

    x1 = center_x - 0.5 * w 
    x2 = center_x + 0.5 * w 
    y1 = center_y - 0.5 * h 
    y2 = center_y + 0.5 * h 

    point_list = []
    point_list.append(rotate_point(x1, y1, center_x, center_y, theta))
    point_list.append(rotate_point(x1, y2, center_x, center_y, theta))
    point_list.append(rotate_point(x2, y2, center_x, center_y, theta))
    point_list.append(rotate_point(x2, y1, center_x, center_y, theta))

    red = (0, 0, 200)
    yellow = (4, 239, 242)
    # Draw rectangle lines (front line in red, others in yellow)
    cv2.line(image, point_list[0], point_list[1], yellow, thickness)
    cv2.line(image, point_list[1], point_list[2], yellow, thickness)
    cv2.line(image, point_list[2], point_list[3], red, thickness)
    cv2.line(image, point_list[3], point_list[0], yellow, thickness)

    # ---------------------------------------------------------------------------------
    # New logic for calculating closest point from origin and annotating:
    # Origin in the BEV is at (x_offset, bev_size) which we must infer from context.
    # We'll extract these from the known pattern used in draw_bev (which calls this).
    # According to draw_bev, x world = (x_pixel - x_offset)/scale and y world = (bev_size - y_pixel)/scale.
    # We do not have direct access to scale and x_offset here, but we can assume they are consistent:
    # Let's find them from global usage. As per the instructions, we have to use current methods and arguments.
    # We'll assume the same defaults: x_offset=250, scale=10, bev_size=500
    # (These were given as arguments in draw_bev and init_bev, let's keep them consistent.)

    x_offset = 250
    scale = 10
    bev_size = 500

    # Convert each corner point to world coordinates
    world_corners = []
    for pt in point_list:
        px, py = pt
        # World coords relative to origin:
        world_x = (px - x_offset) / scale
        world_y = (bev_size - py) / scale
        world_corners.append((world_x, world_y))

    # Find the closest corner to the origin (0,0) in world coords
    distances = [math.sqrt(wx*wx + wy*wy) for (wx, wy) in world_corners]
    closest_idx = int(np.argmin(distances))

    # Mark the closest corner with a green circle
    closest_pt_pixel = point_list[closest_idx]
    cv2.circle(image, closest_pt_pixel, 4, (0, 255, 0), -1)

    # Add text to the closest point using its world coordinates
    wx_closest, wy_closest = world_corners[closest_idx]
    label = f"({wx_closest:.2f}, {wy_closest:.2f}) Meters"
    cv2.putText(image, label, (closest_pt_pixel[0] + 5, closest_pt_pixel[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return (wx_closest, wy_closest)


def draw_bev(image_bev, x3d, z3d, l3d, w3d, ry3d, x_offset=250, scale=10, bev_size=500):
    bev_rect = [0, 0, 0, 0, 0]
    bev_rect[0] = x3d * scale + x_offset
    bev_rect[1] = bev_size - z3d * scale
    bev_rect[2] = l3d * scale
    bev_rect[3] = w3d * scale
    bev_rect[4] = ry3d
    x_close, z_close = draw_bev_rect(image_bev, bev_rect, x3d, z3d)
    return (x_close, z_close)


def init_bev(x_offset=250, scale=10, bev_size=500, dis_interval=5, thickness=2):
    image_bev = np.zeros((bev_size, bev_size, 3), np.uint8)
    # Draw distance circles
    for i in range(1, 20):
        cv2.circle(image_bev, (x_offset, bev_size), scale * i * dis_interval, (255, 255, 255), thickness)

    # ---------------------------------------------------------------------------------
    # Add x and y coordinate lines to the bird's eye view
    # The origin in BEV is at (x_offset, bev_size).
    # Draw the x-axis line (horizontal) passing through the origin:
    cv2.line(image_bev, (0, bev_size), (bev_size, bev_size), (255, 255, 255), 1)
    # Draw the y-axis line (vertical) passing through the origin:
    cv2.line(image_bev, (x_offset, 0), (x_offset, bev_size), (255, 255, 255), 1)

    return image_bev

