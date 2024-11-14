# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:18:55 2024

@author: lenovo
"""

import cv2
import numpy as np
import os
from tkinter import Tk, filedialog

# Function to prompt the user to select a file
def ask_file(prompt: str) -> str:
    Tk().withdraw()
    return filedialog.askopenfilename(title=prompt)

# Mouse callback function for selecting the ROI (Region of Interest)
def on_mouse(event: int, x: int, y: int, flag: int, param: tuple):
    rect, left_button_flags = param
    left_button_down, left_button_up = left_button_flags

    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0], rect[1], rect[2], rect[3] = x, y, x, y
        left_button_flags[0], left_button_flags[1] = True, False

    elif event == cv2.EVENT_MOUSEMOVE and left_button_down and not left_button_up:
        rect[2], rect[3] = x, y

    elif event == cv2.EVENT_LBUTTONUP and left_button_down and not left_button_up:
        x_min, y_min = min(rect[0], rect[2]), min(rect[1], rect[3])
        x_max, y_max = max(rect[0], rect[2]), max(rect[1], rect[3])
        rect[:] = [x_min, y_min, x_max, y_max]
        left_button_flags[0], left_button_flags[1] = False, True

# Mouse callback function for fine-tuning the mask
def draw_2(event: int, x: int, y: int, flag: int, param: dict):
    drawing_flags, imgmask, img, mask, bgdModel, fgdModel, output_dir, img_base_name, img_ext = param.values()
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_flags['bg'] = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing_flags['bg']:
        cv2.circle(imgmask, (x, y), 3, (0, 0, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_flags['bg'] = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_flags['fg'] = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing_flags['fg']:
        cv2.circle(imgmask, (x, y), 3, (255, 255, 255), -1)

    elif event == cv2.EVENT_RBUTTONUP:
        drawing_flags['fg'] = False

    elif event == cv2.EVENT_RBUTTONDBLCLK:
        mask_img_name = f"{img_base_name}_mask{img_ext}"
        file_path = os.path.join(output_dir, mask_img_name)
        cv2.imwrite(file_path, imgmask)  # Save the mask image
        imgmask_BF = cv2.imread(file_path)
        mask3 = cv2.cvtColor(imgmask_BF, cv2.COLOR_BGR2GRAY)
        mask[mask3 == 0], mask[mask3 == 255] = 0, 1
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask3 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img *= mask3[:, :, np.newaxis]
        cv2.imshow('Segmented area', img)

        final_output_path = os.path.join(output_dir, f"{img_base_name}_segment{img_ext}")
        cv2.imwrite(final_output_path, img)
        #print(f"Image saved to {final_output_path}")
        
    cv2.imshow('Draw outline and fill defective area', imgmask)

def main():
    # Ask user to select the input image file
    img_path = ask_file("Select the input image file")
    img = cv2.imread(img_path)
    imgmask = img.copy()
    
    # Get the base name of the input image file and its directory
    img_name = os.path.basename(img_path)
    img_base_name, img_ext = os.path.splitext(img_name)
    output_dir = os.path.dirname(img_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the input image to the output directory
    input_image_output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(input_image_output_path, img)

    # Initialize mask and models for grabCut algorithm
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = [0, 0, 0, 0]
    left_button_flags = [False, True]  # leftButtonDown, leftButtonUp

    # Create the bounding box window
    bounding_window_name = 'bounding_box'
    bounding_window_title = 'Draw a bounding box around the defect'
    cv2.namedWindow(bounding_window_name)
    cv2.setWindowTitle(bounding_window_name, bounding_window_title)
    cv2.setMouseCallback(bounding_window_name, on_mouse, (rect, left_button_flags))
    cv2.imshow(bounding_window_name, img)

    # Main loop to handle the user interaction for ROI selection
    while cv2.waitKey(2) == -1:
        if left_button_flags[0] and not left_button_flags[1]:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow(bounding_window_name, img_copy)

        elif not left_button_flags[0] and left_button_flags[1] and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
            rect[2], rect[3] = rect[2] - rect[0], rect[3] - rect[1]
            rect_copy = tuple(rect.copy())
            rect[:] = [0, 0, 0, 0]
            cv2.grabCut(img, mask, rect_copy, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Initialize drawing flags
            drawing_flags = {'bg': False, 'fg': False}
            param = {
                'drawing_flags': drawing_flags,
                'imgmask': imgmask,
                'img': img,
                'mask': mask,
                'bgdModel': bgdModel,
                'fgdModel': fgdModel,
                'output_dir': output_dir,
                'img_base_name': img_base_name,
                'img_ext': img_ext
            }

            outline_filling_window_name = 'Draw outline and fill defective area'
            cv2.namedWindow(outline_filling_window_name)
            cv2.setMouseCallback(outline_filling_window_name, draw_2, param)
            cv2.imshow(outline_filling_window_name, imgmask)

    final_output_path = os.path.join(output_dir, f"{img_base_name}_segment{img_ext}")
    cv2.imwrite(final_output_path, img)
    #print(f"{final_output_path}")

    # Return the input image and final output image paths via stdout
    print(f"{img_path},{final_output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()