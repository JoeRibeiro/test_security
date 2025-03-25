simport json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
import cv2
import numpy as np
import itertools
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import tkinter as tk
from tkinter import simpledialog
import logging
from scipy.stats import gaussian_kde
import csv

logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

import sqlite3

secretkey='sv=2021-04-10&ss=b&srt=sco&sp=rwdlacx&se=2025-03-25T12:00:00Z&st=2025-03-25T10:00:00Z&spr=https&sig=abcdefghijklmnopqrstuvwxyz1234567890'


username = input("Enter your username: ")
password = input("Enter your password: ")


class_mapping = {'gauge': 0, 'CREunknown': 1}

def angle_between_points(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * (180.0 / np.pi)
    return angle

def address_gauges_with_more_than_four_points(points, eps=50, min_samples=1):
    if len(points) <= 4:
        #print("Returning original points as they are 4 or fewer.")
        return points    
    points_array = np.array(points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
    labels = db.labels_
    centroids = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label != -1:
            cluster_points = points_array[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
    centroids = np.array(centroids)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    labels = db.labels_
    new_centroids = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label != -1:
            cluster_points = centroids[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(centroid)
    centroids = np.array(new_centroids)
    angles = []
    n = len(centroids)
    for i in range(n):
        p1 = centroids[i - 1] if i > 0 else centroids[n - 1]
        p2 = centroids[i]
        p3 = centroids[i + 1] if i < n - 1 else centroids[0]
        angle = angle_between_points(p1, p2, p3)
        angles.append(angle)
    angle_differences = [abs(angle - 180) for angle in angles]
    sorted_indices = np.argsort(angle_differences)[::-1]
    important_points = centroids[sorted_indices[:4]]
    return important_points.tolist()

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roid_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def visualize_predictions_with_contours(model, image, device, class_mapping, dilation_iterations=1):
    model.eval()
    with torch.no_grad():
        predictions = model(image)
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    overlay_img = img_np.copy()
    all_contours = []
    kernel = np.ones((8, 8), np.uint8) 
    for idx, mask in enumerate(predictions[0]['masks']):
        if predictions[0]['scores'][idx] > 0.9:
            mask_np = mask[0].cpu().numpy()
            mask_np = (mask_np > 0.9).astype(np.uint8)
            dilated_mask = cv2.dilate(mask_np, kernel, iterations=dilation_iterations)
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
            for contour in contours:
                cv2.drawContours(overlay_img, [contour], -1, (0, 255, 0), 2)
            xmin, ymin, xmax, ymax = predictions[0]['boxes'][idx].cpu().numpy()
            label_idx = predictions[0]['labels'][idx].item()
            label = list(class_mapping.keys())[list(class_mapping.values()).index(label_idx)] if label_idx in class_mapping.values() else "Unknown"
            score = predictions[0]['scores'][idx].item()
            cv2.rectangle(overlay_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
            cv2.putText(overlay_img, f"{label}: {score:.2f}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    print(f"Number of machine learning contours detected: {len(all_contours)}")
    return overlay_img, predictions, all_contours
    
num_classes = len(class_mapping) + 1
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load('outputs/crab_rcnn_model.pth'), strict=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_closest_pairs(points):
    distances = []
    for (i, point1), (j, point2) in itertools.combinations(enumerate(points), 2):
        dist = calculate_distance(point1, point2)
        distances.append((dist, i, j))
    distances.sort()
    pairs = []
    used_indices = set()
    for dist, i, j in distances:
        if i not in used_indices and j not in used_indices:
            pairs.append((i, j))
            used_indices.update([i, j])
    return pairs

def assign_corners(points, pairs):
    pair_means = []
    for i, j in pairs:
        mean_x = (points[i][0] + points[j][0]) / 2
        mean_y = (points[i][1] + points[j][1]) / 2
        pair_means.append((mean_x, mean_y, i, j))
    pair_means.sort(key=lambda x: x[0])
    left_pair = pair_means[0]
    right_pair = pair_means[1]
    if points[left_pair[2]][1] < points[left_pair[3]][1]:
        upper_left, lower_left = left_pair[2], left_pair[3]
    else:
        upper_left, lower_left = left_pair[3], left_pair[2]
    if points[right_pair[2]][1] < points[right_pair[3]][1]:
        upper_right, lower_right = right_pair[2], right_pair[3]
    else:
        upper_right, lower_right = right_pair[3], right_pair[2]
    src_points = {
        "upper_left": points[upper_left],
        "upper_right": points[upper_right],
        "lower_right": points[lower_right],
        "lower_left": points[lower_left]
    }
    try:
        polygon = Polygon(list(src_points.values()))
        if not polygon.is_valid:
            #print("Invalid polygon detected. Reordering.")            
            src_points = {
                "upper_left": points[upper_left],
                "upper_right": points[lower_right],
                "lower_right": points[upper_right],
                "lower_left": points[lower_left]
            }            
    except Exception as e:
        print(f"Error creating polygon: {e}")
        return None
    return src_points

def transform_polygons(polygons, matrix):
    transformed_polygons = []
    for polygon in polygons:
        points = np.array(polygon, dtype=np.float32)
        transformed_points = cv2.perspectiveTransform(np.array([points]), matrix)
        transformed_points = transformed_points[0].tolist()
        transformed_points.append(transformed_points[0])
        transformed_polygons.append(transformed_points)
    return transformed_polygons

def transform_image(image, corners):
    src_points = np.array([
        corners["upper_left"],
        corners["upper_right"],
        corners["lower_right"],
        corners["lower_left"]
    ], dtype=np.float32)
    pad = 4000
    dst_points = np.array([
        [0 + pad, 0 + pad],
        [rectangle_x_length + pad, 0 + pad],
        [rectangle_x_length + pad, rectangle_y_length + pad],
        [0 + pad, rectangle_y_length + pad]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(np.array(image), matrix, (image.width+8000, image.height+8000))
    return transformed_image, matrix

def check_self_intersection(points):
    polygon = Polygon(points)
    return polygon.is_valid

def calculate_long_axis(polygon):
    max_distance = 0
    axis_points = (None, None)
    for i, point1 in enumerate(polygon):
        for point2 in polygon[i+1:]:
            dist = calculate_distance(point1, point2)
            if dist > max_distance:
                max_distance = dist
                axis_points = (point1, point2)
    return max_distance, axis_points

def adjust_annotations_after_crop(polygons, crop_bounds):
    min_row, max_row, min_col, max_col = crop_bounds
    adjusted_polygons = []
    for polygon in polygons:
        adjusted_polygon = [
            (x - min_col, y - min_row) for (x, y) in polygon
        ]
        if len(adjusted_polygon) >= 4:
            poly_area = Polygon(adjusted_polygon).area
            if poly_area > filtersmallerthan:
                adjusted_polygons.append(adjusted_polygon)
    return adjusted_polygons

def trim_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    non_background_pixels = np.where(thresholded > 0)
    min_row = np.min(non_background_pixels[0])
    max_row = np.max(non_background_pixels[0])
    min_col = np.min(non_background_pixels[1])
    max_col = np.max(non_background_pixels[1])
    cropped_image = image[min_row:max_row, min_col:max_col]
    return cropped_image, (min_row, max_row, min_col, max_col)


def plot_annotations_with_corners(image_path, annotations, model, device, class_mapping, overlap_log):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image)
    polygons_to_transform = []
    gauge_corners = None
    long_axes = []
    for annotation in annotations:
        we_have_a_pair = False  
        for result in annotation['result']:
            if result['type'] == 'polygonlabels':
                points = result['value']['points']
                label = result['value']['polygonlabels'][0]
                measurement = result['value']['measurement'][0] if 'measurement' in result.get('value', {}) and result['value']['measurement'] else ''
                color = label_colors.get(label, 'r')
                polygon_points = [
                    (x * image.width / 100, y * image.height / 100)
                    for x, y in points
                ]
                polygon = patches.Polygon(
                    polygon_points, closed=True, edgecolor=color, facecolor='none', linewidth=2
                )
                ax1.add_patch(polygon)
                if label.lower() == "gauge":
                    simplified_points = address_gauges_with_more_than_four_points(polygon_points)
                    for corner in simplified_points:
                        ax1.plot(
                            corner[0], corner[1], 'o', color='white', markersize=8, markeredgecolor='black'
                        )
                    closest_pairs = find_closest_pairs(simplified_points)
                    gauge_corners = assign_corners(simplified_points, closest_pairs)
                polygons_to_transform.append((polygon_points, label, measurement))
    test_image = get_transform(train=False)(image).unsqueeze(0).to(device)
    overlay_img, predictions, cnncontours = visualize_predictions_with_contours(model, test_image, device, class_mapping)
    cnncontours = [[tuple(point[0]) for point in contour] for contour in cnncontours]
    ax1.imshow(overlay_img, alpha=0.5) 
    ax1.set_title('Original image. Blue = annotated, green = predicted.')
    if gauge_corners:
        transformed, matrix = transform_image(image, gauge_corners)
        transformed_polygons = transform_polygons([poly for poly, _, _ in polygons_to_transform], matrix)
        transformed_cnncontours = transform_polygons(cnncontours, matrix)
        cropped_transformed, crop_bounds = trim_image(transformed)
        adjusted_polygons = adjust_annotations_after_crop(transformed_polygons, crop_bounds)
        adjusted_cnncontours = adjust_annotations_after_crop(transformed_cnncontours, crop_bounds)        
        ax2.imshow(cropped_transformed)
        for adjusted_cnncontour in adjusted_cnncontours:
            adjusted_cnncontour_polygon = Polygon(adjusted_cnncontour)
            adjusted_cnncontour_polygon = adjusted_cnncontour_polygon.buffer(0.01)
            max_overlap_percentage = 0
            matching_polygon_area = 0
            matching_polygon_label = None
            for adjusted_polygon, label, measurement in zip(adjusted_polygons, [label for _, label, _ in polygons_to_transform], [measurement for _, _, measurement in polygons_to_transform]):
                adjusted_polygon_polygon = Polygon(adjusted_polygon)
                adjusted_polygon_polygon = adjusted_polygon_polygon.buffer(0.01)        
                intersection_area = adjusted_cnncontour_polygon.intersection(adjusted_polygon_polygon).area
                union_area = adjusted_cnncontour_polygon.union(adjusted_polygon_polygon).area
                overlap_percentage = (intersection_area / union_area) * 100
                matching_polygon_measurement = None
                if overlap_percentage > max_overlap_percentage:
                    max_overlap_percentage = overlap_percentage
                    matching_polygon_area = adjusted_polygon_polygon.area
                    matching_polygon_label = label
                    matching_polygon_measurement = measurement
            # Log the max_overlap_percentage for the current image and polygon
            if image_path not in overlap_log:
                overlap_log[image_path] = []
            overlap_log[image_path].append(max_overlap_percentage)
            adjusted_cnncontour_np = np.array(adjusted_cnncontour, dtype=np.int32)
            ax2.plot(
                adjusted_cnncontour_np[:, 0],
                adjusted_cnncontour_np[:, 1],
                color='green',
                linewidth=2
            )
            long_axis_length, axis_points = calculate_long_axis(adjusted_cnncontour_np)
            long_axes.append(long_axis_length)
            point1, point2 = axis_points
            ax2.plot([point1[0], point2[0]], [point1[1], point2[1]], color='orange', linewidth=2, linestyle='--')
            ax2.scatter([point1[0], point2[0]], [point1[1], point2[1]], color='grey', zorder=5)
            if matching_polygon_measurement:
                ax2.text((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2,  f"CNN length: {long_axis_length:.2f} mm, actually {matching_polygon_measurement} mm",  color='white', fontsize=8, ha='center', va='center'            )
            else:
                ax2.text((point1[0] + point2[0]) / 2, (point1[1] + point2[1]), f"CNN length: {long_axis_length:.2f} mm", color='white', fontsize=8, ha='center', va='center')
            if long_axis_length and matching_polygon_measurement:
                we_have_a_pair = True
                long_axis_est = long_axis_length
                measuredvalue = matching_polygon_measurement
        for adjusted_polygon in adjusted_polygons:
            adjusted_polygon_np = np.array(adjusted_polygon, dtype=np.int32)
            ax2.plot(
                adjusted_polygon_np[:, 0],
                adjusted_polygon_np[:, 1],
                color='blue',
                linewidth=2
            )
            long_axis_lengthc, axis_points = calculate_long_axis(adjusted_polygon_np)
            long_axes.append(long_axis_lengthc)
            point1, point2 = axis_points
            ax2.plot([point1[0], point2[0]], [point1[1], point2[1]], color='yellow', linewidth=2, linestyle='--')
            ax2.scatter([point1[0], point2[0]], [point1[1], point2[1]], color='red', zorder=5)
    ax2.set_title('Quadrat-transformed image with measurements.')            
    ax1.axis('off')
    ax2.axis('off')
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()


wdir = 'C:/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/crabcv/'
os.chdir(wdir)

with open('labels/Kate_and_Dan_Labelledproject-1-at-2025-01-29-13-23-42670be8_cleaned_modified.json', 'r') as file:
    data = json.load(file)

label_colors = {
    "CREunknown": "b",
    "CREmale": "p",
    "CREMalePartial": "g",
    "CREfemale": "m",
    "Gauge": "y",
    "CREUnknownPartial": "c"
}

rectangle_x_length = 400 
rectangle_y_length = 200 
filtersmallerthan = 100

# Initialize the dictionary to log overlap percentages
overlap_log = {}

for entry in data:
    imagefile = entry['data']['filenamecleaned']
    if "Cefas_Northeast" not in imagefile:
        continue    
    image_path = os.path.join('images_all_including_lobster', imagefile)
    if os.path.exists(image_path):
        plot_annotations_with_corners(image_path, entry['annotations'], model, device, class_mapping, overlap_log)
    else:
        print(f"Image file {image_path} not found.")

print(overlap_log)


# Extract overlap percentages from the dictionary
overlap_percentages = list(itertools.chain(*overlap_log.values()))

# Remove values below 10%
filtered_percentages = [value for value in overlap_percentages if value >= 10]

# Calculate statistics
mean_overlap = np.mean(filtered_percentages)
median_overlap = np.median(filtered_percentages)
quartiles = np.percentile(filtered_percentages, [25, 50, 75])
min_overlap = np.min(filtered_percentages)
max_overlap = np.max(filtered_percentages)
std_dev_overlap = np.std(filtered_percentages)

# Print the statistics
print(f"Mean Overlap: {mean_overlap:.2f}%")
print(f"Median Overlap: {median_overlap:.2f}%")
print(f"Quartiles: {quartiles}")
print(f"Min Overlap: {min_overlap:.2f}%")
print(f"Max Overlap: {max_overlap:.2f}%")
print(f"Standard Deviation: {std_dev_overlap:.2f}%")

# Create a density plot using Gaussian Kernel Density Estimation
density = gaussian_kde(overlap_percentages, bw_method=0.01)
x = np.linspace(min(overlap_percentages), max(overlap_percentages), 1000)
y = density(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, color='g')

# Add labels and title
plt.xlabel('Overlap Percentage')
plt.ylabel('Density')
plt.title('Density Plot of Overlap Percentages')

# Show the plot
plt.show()

with open('overlap_percentages.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Overlap Percentage'])
    for percentage in overlap_percentages:
        writer.writerow([percentage])


x_values = []
y_values = []
with open('outputs/saved_pairs.txt', 'r') as file:
    for line in file:
        parts = line.split(',')
        x_values.append(float(parts[0]))
        y_values.append(float(parts[1]))

# Plot the scatter graph
plt.scatter(x_values, y_values)

# Add a 1:1 line
min_val = min(min(x_values), min(y_values))
max_val = max(max(x_values), max(y_values))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Pairing of CNN measurement and calipers')
plt.show()
