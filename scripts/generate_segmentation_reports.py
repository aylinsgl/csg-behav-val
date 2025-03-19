"""
Scene Segmentation Report Generator
==========================================
Author: Aylin Kallmayer
Last updated: 2025-03-19
==========================================
Description:
This script generates segmentation reports for ADE20K and SCEGRAM datasets.
It loads JSON files, creates polygons from coordinate data, finds touching objects,
and consolidates segmentation reports. It supports both the ADEK20K and SCEGRAM datasets,
providing functions for generating segmentation reports using different methods and 
merging them into comprehensive CSV reports.

Usage:
    Run the script with the desired imageset and trainset flag to generate the segmentation report.
    For example:
        python segmentation_reports.py
"""

import os
import json
import pandas as pd
import csv
from shapely.geometry import Polygon
from csg import utils  # Assumes utils contains functions like calculate_centroid_x_y, load_mask, calculate_centroid_mask

def load_json(file_path):
    """
    Load the JSON file with proper encoding handling.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            json_data = json.load(file)
    return json_data

def create_polygon_from_coordinates(polygon_data):
    """
    Convert the polygon x, y coordinates into a Shapely polygon.
    
    Parameters:
        polygon_data (dict): Dictionary containing 'x' and 'y' coordinate lists.
    
    Returns:
        Polygon or None: A valid Shapely polygon, or None if insufficient points.
    """
    points = list(zip(polygon_data['x'], polygon_data['y']))  # List of (x, y) tuples
    
    if len(points) < 3:
        print(f"Polygon has too few points: {points}")
        return None
    
    if points[0] != points[-1]:
        points.append(points[0])
    
    polygon = Polygon(points)

    if not polygon.is_valid:
        print(f"Invalid polygon found, attempting to fix: {points}")
        polygon = polygon.buffer(0)
    
    return polygon

def find_touching_objects(polygons):
    """
    Find and return a dictionary mapping object IDs to a list of object IDs that are touching.

    Parameters:
        polygons (dict): Dictionary mapping object IDs to Shapely Polygon objects.

    Returns:
        dict: Dictionary with object IDs as keys and lists of touching object IDs as values.
    """
    touching_dict = {obj_id: [] for obj_id in polygons.keys()}
    object_ids = list(polygons.keys())
    for i in range(len(object_ids)):
        for j in range(i + 1, len(object_ids)):
            obj_id_1 = object_ids[i]
            obj_id_2 = object_ids[j]
            try:
                if polygons[obj_id_1] and polygons[obj_id_2] and polygons[obj_id_1].touches(polygons[obj_id_2]):
                    touching_dict[obj_id_1].append(obj_id_2)
                    touching_dict[obj_id_2].append(obj_id_1)
            except Exception as e:
                print(f"Error processing polygons {obj_id_1} and {obj_id_2}: {e}")
    return touching_dict

def segmentation_adek20k(category_a, category_b_list, depth_scale_factor=100, trainset=True):
    """
    Generate segmentation reports for the ADEK20K dataset.

    Parameters:
        category_a (str): Primary category folder name.
        category_b_list (list): List of sub-category folder names.
        depth_scale_factor (int, optional): Scaling factor for depth values.
        trainset (bool, optional): Flag indicating whether to use the training set.
    """
    for category_b in category_b_list:
        if trainset:
            folder_path = f'/Volumes/Volume/RonYas/ADE20K_2021_17_01/images/ADE/training/{category_a}/{category_b}/'
        else:
            folder_path = f'/Volumes/Volume/RonYas/ADE20K_2021_17_01/images/ADE/validation/{category_a}/{category_b}/'
        data = []
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    json_data = load_json(file_path)
                    folder = json_data['annotation']['folder']
                    filename = json_data['annotation']['filename']
                    image_width = json_data['annotation']['imsize'][1]
                    image_height = json_data['annotation']['imsize'][0]
                    scene_category = json_data['annotation']['scene'][-1]
                    
                    polygons = {}
                    objects = json_data['annotation']['object']
                    
                    for idx, obj in enumerate(objects):
                        new_id = idx
                        polygons[new_id] = create_polygon_from_coordinates(obj['polygon'])
                        obj['new_id'] = new_id
                    
                    touching_dict = find_touching_objects(polygons)
                    
                    for obj in objects:
                        object_name = obj['name']
                        polygon_x = obj['polygon']['x']
                        polygon_y = obj['polygon']['y']
                        depth_ordering_rank = obj.get('depth_ordering_rank', 0)
                        new_id = obj['new_id']
                        centroid_y, centroid_x = utils.calculate_centroid_x_y(polygon_x, polygon_y)
                        scaled_depth = depth_ordering_rank * depth_scale_factor / max(image_width, image_height)
                        touching_objects = touching_dict.get(new_id, [])
                        data.append({
                            'object_id': new_id,
                            'filename': filename,
                            'folder': folder,
                            'scene_category': scene_category,
                            'object_name': object_name,
                            'centroid_x': centroid_x,
                            'centroid_y': centroid_y,
                            'centroid_depth': scaled_depth,
                            'touching_objects': touching_objects,
                            'image_width': image_width,
                            'image_height': image_height
                        })
        df = pd.DataFrame(data)
        print(df.head())
        output_dir = 'results/segmentation_reports/ADEK20K'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f'{output_dir}/{category_b}_segmentation_report-trainset_{trainset}.csv', index=False)

def segmentation_adek20k_old(category_a, category_b_list, depth_scale_factor=100):
    """
    Generate segmentation reports for the ADEK20K dataset using the old method.

    Parameters:
        category_a (str): Primary category folder name.
        category_b_list (list): List of sub-category folder names.
        depth_scale_factor (int, optional): Scaling factor for depth values.
    """
    for category_b in category_b_list:
        folder_path = f'/Volumes/Volume/RonYas/ADE20K_2021_17_01/images/ADE/training/{category_a}/{category_b}/'
        data = []
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            json_data = json.load(file)
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='ISO-8859-1') as file:
                            json_data = json.load(file)
                    folder = json_data['annotation']['folder']
                    filename = json_data['annotation']['filename']
                    image_width = json_data['annotation']['imsize'][1]
                    image_height = json_data['annotation']['imsize'][0]
                    scene_category = json_data['annotation']['scene'][-1]
                    for obj in json_data['annotation']['object']:
                        object_name = obj['name']
                        polygon_x = obj['polygon']['x']
                        polygon_y = obj['polygon']['y']
                        depth_ordering_rank = obj.get('depth_ordering_rank', 0)
                        centroid_y, centroid_x = utils.calculate_centroid_x_y(polygon_x, polygon_y)
                        scaled_depth = depth_ordering_rank * depth_scale_factor / max(image_width, image_height)
                        centroid_3d = (centroid_x, centroid_y, scaled_depth)
                        data.append({
                            'filename': filename,
                            'folder': folder,
                            'scene_category': scene_category,
                            'object_name': object_name,
                            'centroid_x': centroid_3d[0],
                            'centroid_y': centroid_3d[1],
                            'centroid_depth': centroid_3d[2],
                            'image_width': image_width,
                            'image_height': image_height
                        })
        df = pd.DataFrame(data)
        print(df.head())
        output_dir = 'results/segmentation_reports/ADEK20K'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f'{output_dir}/{category_b}_segmentation_report.csv', index=False)

def consolidate_adek20k(files, trainset=True):
    """
    Consolidate multiple ADEK20K segmentation report CSV files into a single CSV file.

    Parameters:
        files (list): List of CSV filenames to consolidate.
        trainset (bool, optional): Whether the reports are from the training set.
    """
    data_frames = []
    for csv_file in files:
        file_path = os.path.join('results/segmentation_reports/ADEK20K/', csv_file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    consolidated_df = pd.concat(data_frames, ignore_index=True)
    consolidated_df.to_csv(f'results/segmentation_reports/ADEK20K/consolidated_segmentation_report-trainset_{trainset}.csv', index=False)

def segmentation_scegram(image_dir, output_csv):
    """
    Generate segmentation reports for the SCEGRAM dataset.

    Parameters:
        image_dir (str): Directory containing the images.
        output_csv (str): Path to the output CSV file.
    """
    filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.startswith('.')]
    data = []
    for f in filenames:
        filename = f
        scene_id = f.split('Scene')[1].split('_')[0]
        consistency = f.split('_')[1]
        object_name = '_'.join(f.split('_')[2:]).replace('.png', '')
        mask = utils.load_mask(os.path.join(image_dir, f))
        centroid = utils.calculate_centroid_mask(mask)
        data.append([filename, scene_id, consistency, object_name, centroid[1], centroid[0]])
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'scene_id', 'consistency', 'object_name', 'centroid_x', 'centroid_y'])
        writer.writerows(data)
    print(f"CSV file '{output_csv}' created successfully with centroid coordinates.")

def main(imageset, trainset=True):
    """
    Main function to generate segmentation reports for a specified dataset.

    Parameters:
        imageset (str): Dataset to process ('adek20k' or 'scegram').
        trainset (bool, optional): Flag indicating whether to process training set data.
    """
    if imageset == 'adek20k':
        segmentation_adek20k('home_or_hotel', ['bathroom','bedroom','dining_room','kitchen','living_room'], trainset=trainset)
        files = [f"living_room_segmentation_report-trainset_{trainset}.csv",
                 f"kitchen_segmentation_report-trainset_{trainset}.csv",
                 f"dining_room_segmentation_report-trainset_{trainset}.csv",
                 f"bedroom_segmentation_report-trainset_{trainset}.csv",
                 f"bathroom_segmentation_report-trainset_{trainset}.csv"]
        consolidate_adek20k(files, trainset=trainset)
    elif imageset == 'scegram':
        segmentation_scegram('/Volumes/Extreme SSD/Langmark_Deploy_08_Aug_2024_19_01_27/', 
                             'results/segmentation_reports/SCEGRAM/consolidated_segmentation_report.csv')

if __name__ == '__main__':
    main('adek20k')
