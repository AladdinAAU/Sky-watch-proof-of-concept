import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import torch
from LightGlue.lightglue.lightglue import LightGlue
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d, SuperPoint
from tqdm import tqdm
import time
from pathlib import Path
import math
import torchvision.transforms.functional as TF
import logging

# -----------------------------------------------------
# 1. LOGGING CONFIGURATION
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------------------------------
# 2. UTILITY FUNCTIONS
# -----------------------------------------------------

def rotate_tensor_image(image_tensor, angle_degrees):
    """Rotate a tensor image by the given angle using bilinear interpolation."""
    rotated_tensor = TF.rotate(image_tensor, angle=angle_degrees, interpolation=TF.InterpolationMode.BILINEAR)
    return rotated_tensor

def load_yaw_angles(csv_file):
    """Loads yaw angles from a CSV file (radians converted to degrees)."""
    yaw_angles = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            yaw_rad = float(row['yaw'])
            yaw_deg = math.degrees(yaw_rad)
            yaw_angles[filename] = yaw_deg
    logging.info(f"Loaded yaw angles for {len(yaw_angles)} images from {csv_file}.")
    return yaw_angles

def setup_device():
    """Determines whether to use GPU (if available) or CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using GPU.")
    else:
        logging.info("CUDA is not available. Using CPU.")
    logging.info(f"Using device: {device}")
    return device

def split_image_into_tiles(image, tile_size=(512, 512), overlap=128, roi=None):
    """
    Splits an image tensor [C, H, W] into tiles of size tile_size with the specified overlap.
    Optionally restricts splitting to an ROI defined as (x_min, y_min, x_max, y_max).
    Returns a list of tuples: (tile_tensor, x, y), where (x,y) is the tile’s top-left coordinate.
    """
    _, H, W = image.shape
    tile_h, tile_w = tile_size
    stride_x = tile_w - overlap
    stride_y = tile_h - overlap

    if roi is None:
        x_min, y_min, x_max, y_max = 0, 0, W, H
    else:
        x_min, y_min, x_max, y_max = roi
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(W, x_max), min(H, y_max)

    tiles = []
    for y in range(y_min, y_max, stride_y):
        for x in range(x_min, x_max, stride_x):
            current_tile_h = min(tile_h, y_max - y)
            current_tile_w = min(tile_w, x_max - x)
            if current_tile_h <= 0 or current_tile_w <= 0:
                continue
            tile = image[:, y:y+current_tile_h, x:x+current_tile_w]
            tiles.append((tile, x, y))
    return tiles

def visualize_tile_grid(tiles, satellite_image_path, tile_size=(512,512)):
    """
    Overlays tile bounding boxes onto the satellite image for debugging.
    Uses the same tile_size as in splitting.
    """
    sat_img_color = cv2.cvtColor(cv2.imread(satellite_image_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12,12))
    plt.imshow(sat_img_color)
    tile_h, tile_w = tile_size
    for idx, (tile, x, y) in enumerate(tiles):
        x_min, y_min = x, y
        x_max, y_max = x + tile_w, y + tile_h
        plt.plot([x_min, x_max, x_max, x_min, x_min],
                 [y_min, y_min, y_max, y_max, y_min],
                 'r', linewidth=1)
        plt.text(x_min, y_min, f'Tile {idx}', color='yellow', fontsize=8)
    plt.title("Tile Grid on Satellite Image")
    plt.axis('off')
    plt.savefig('tile_grid.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def initialize_models(device, max_keypoints=2048):
    """Initializes the SuperPoint extractor and LightGlue matcher models."""
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    logging.info("Models initialized.")
    return extractor, matcher

def extract_features(extractor, images, device):
    """
    Extract features for a set of images (drone images).
    Returns a dictionary mapping image names to their feature dictionaries.
    """
    features = {}
    with torch.no_grad():
        for name, img in images.items():
            img = img.to(device)
            feats = extractor.extract(img.unsqueeze(0))
            feats = rbd(feats)  # Remove batch dimension
            features[name] = feats
    logging.info("Feature extraction for drone images completed.")
    return features

def extract_features_from_tiles_and_cache(tiles, extractor, device, sat_image_name):
    """
    Extracts features for each satellite tile and caches the results.
    Stores keypoints in local tile coordinates (no offset added here).
    """
    features_list = []
    feats_dir = Path('cached_features') / sat_image_name.replace('.', '_')
    feats_dir.mkdir(parents=True, exist_ok=True)

    for tile, x, y in tqdm(tiles, desc=f"Extracting features for tiles of {sat_image_name}"):
        cache_filename = feats_dir / f"tile_{x}_{y}_features.pt"
        if cache_filename.exists():
            feats = torch.load(cache_filename)
            logging.info(f"Loaded cached features for tile ({x}, {y}).")
        else:
            tile_tensor = tile.unsqueeze(0).to(device)
            with torch.no_grad():
                raw_feats = extractor.extract(tile_tensor)
                # Remove batch dimension and flip keypoints from (y,x) to (x,y)
                feats = {
                    "keypoints": raw_feats["keypoints"].squeeze(0)[:, [1, 0]],
                    "descriptors": raw_feats["descriptors"].squeeze(0)
                }
            torch.save(feats, cache_filename)
            logging.info(f"Cached features for tile ({x}, {y}) at {cache_filename}.")
        # Do not add tile offset here; leave keypoints as local coordinates.
        features_list.append({
            'tile_position': (x, y),
            'features': feats
        })
    return features_list

def match_features(matcher, feats0, feats1, device):
    """
    Matches features from feats0 and feats1 using LightGlue.
    Prepares feature dictionaries by adding a batch dimension.
    """
    required_keys = ['keypoints', 'descriptors']
    for feats, name in zip([feats0, feats1], ['feats0', 'feats1']):
        for key in required_keys:
            if key not in feats:
                raise AssertionError(f"'{key}' not found in {name}.")
            if not isinstance(feats[key], torch.Tensor):
                raise AssertionError(f"'{key}' in {name} must be a torch.Tensor")
        if feats['keypoints'].ndim != 2 or feats['keypoints'].shape[1] != 2:
            raise AssertionError(f"keypoints in {name} should be [N,2]")
    def prepare_features(feats_dict):
        prepared = {}
        for k, v in feats_dict.items():
            prepared[k] = v.float().unsqueeze(0).to(device)
        return prepared
    data = {
        "image0": prepare_features(feats0),
        "image1": prepare_features(feats1)
    }
    try:
        with torch.no_grad():
            out = matcher(data)
    except Exception as e:
        raise RuntimeError(f"Feature matching failed: {e}")
    processed_matches = {k: (v.squeeze(0) if torch.is_tensor(v) else v) for k, v in out.items()}
    return processed_matches

def load_images(images_dir, image_names):
    """
    Loads images from disk using LightGlue’s load_image (which returns a tensor).
    """
    images_path = Path(images_dir)
    loaded_images = {}
    for name in image_names:
        img_path = images_path / name
        if not img_path.exists():
            raise FileNotFoundError(f"Image '{name}' not found in {images_dir}.")
        loaded_images[name] = load_image(str(img_path))
    logging.info(f"Images loaded from {images_dir}.")
    return loaded_images

def load_bonuds(file):
    """
    Loads boundary coordinates from a CSV/text file.
    Expected format: each row has an index and a value.
    """
    with open(file) as csv_file:
        ann = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            ann.append(float(row[1]))
        return ann

def xy_to_coords(boundaries, sat_res, feature_coords):
    """
    Converts pixel coordinates (x,y) into geographic coordinates (lon, lat, alt=0).
    boundaries: [north, south, east, west]
    sat_res: [height, width] of the satellite image.
    """
    north_south = abs(boundaries[0] - boundaries[1])
    east_west = abs(boundaries[2] - boundaries[3])
    px_lon_lat = [east_west / sat_res[1], north_south / sat_res[0]]
    loc_lon_lat = []
    for pt in feature_coords:
        lon = boundaries[3] + (pt[0] * px_lon_lat[0])
        lat = boundaries[0] - (pt[1] * px_lon_lat[1])
        loc_lon_lat.append([lon, lat, 0])
    return loc_lon_lat

def is_within_bounds(cam, boundaries):
    """Checks if a camera coordinate (lon, lat, alt) is within the given boundaries."""
    lon, lat, _ = cam
    north, south, east, west = boundaries
    return (south < lat < north) and (west < lon < east)

# -----------------------------------------------------
# PnP SOLVER AND POSE ESTIMATION
# -----------------------------------------------------

class PnP:
    def __init__(self, camera_matrix, dist_coeffs,
                 ransac=True,
                 ransac_iterations_count=3000,
                 ransac_reprojection_error=5.0):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ransac = ransac
        self.ransac_iterations_count = ransac_iterations_count
        self.ransac_reprojection_error = ransac_reprojection_error

    def pnp(self, object_points_list, image_points_list):
        cam_coords = []
        for i, obj_pts in enumerate(object_points_list):
            img_pts = image_points_list[i]
            if obj_pts.shape[0] < 4:
                cam_coords.append([0, 0, 0])
                continue
            if self.ransac:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, 
                    self.camera_matrix, self.dist_coeffs,
                    iterationsCount=self.ransac_iterations_count,
                    reprojectionError=self.ransac_reprojection_error
                )
            else:
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, 
                    self.camera_matrix, self.dist_coeffs
                )
            if not success:
                cam_coords.append([0, 0, 0])
                continue
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            cam_pos = -rotation_matrix.T @ tvec
            cam_coords.append(cam_pos.flatten())
        return cam_coords

def estimate_camera_pose(clusters, boundaries, sat_res, pnp_solver, last_known_position):
    """
    For each cluster, computes the camera pose using PnP.
    Returns a dictionary with cluster confidence data.
    """
    cluster_confidence = {}
    def haversine_dist(a, b):
        R = 6371000.0
        lon1, lat1 = a[0], a[1]
        lon2, lat2 = b[0], b[1]
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = (math.sin(dlat/2)**2 +
             math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(h), math.sqrt(1-h))
        return R * c

    for label, items in clusters.items():
        if len(items) < 4:
            continue
        sat_pts = np.array([it['sat_pt'] for it in items])
        drone_pts = np.array([it['drone_pt'] for it in items])
        geo_coords = xy_to_coords(boundaries, sat_res, sat_pts)
        obj_pts = np.array(geo_coords, dtype=np.float32).reshape(-1, 1, 3)
        img_pts = np.array(drone_pts, dtype=np.float32).reshape(-1, 1, 2)
        cams = pnp_solver.pnp([obj_pts], [img_pts])
        cam = cams[0]
        cluster_confidence[label] = {
            'cam_position': cam,
            'num_points': len(items)
        }
        las = [last_known_position[0], last_known_position[1]]
        dist2last = haversine_dist(cam, las)
        logging.info(f"Cluster {label}: {cam}, #Points={len(items)}, Dist2Last= {dist2last:.2f}m")
    return cluster_confidence

def select_best_cluster(cluster_confidence, boundaries, last_known_position=None):
    """Selects the best cluster (with most points and valid bounds)."""
    valid_clusters = []
    for label, info in cluster_confidence.items():
        cam_pos = info['cam_position']
        if np.allclose(cam_pos, [0, 0, 0]):
            continue
        if not is_within_bounds(cam_pos, boundaries):
            continue
        valid_clusters.append((label, info))
    if len(valid_clusters) == 0:
        logging.info("No valid clusters within bounds.")
        return None, None
    valid_clusters.sort(key=lambda x: x[1]['num_points'], reverse=True)
    best_label = valid_clusters[0][0]
    best_cam_pos = valid_clusters[0][1]['cam_position']
    return best_label, best_cam_pos

# -----------------------------------------------------
# CLUSTERING & MATCH VISUALIZATION FUNCTIONS
# -----------------------------------------------------

def visualize_clusters_final(sat_kpts, drone_kpts, scores, satellite_image_path, plot=False, output_prefix=''):
    """
    Clusters the satellite keypoints using DBSCAN and optionally plots them.
    Returns a dictionary mapping cluster labels to clustered keypoints.
    """
    sat_image_original_color = cv2.cvtColor(cv2.imread(str(satellite_image_path)), cv2.COLOR_BGR2RGB)
    db = DBSCAN(eps=450, min_samples=5).fit(sat_kpts)
    labels = db.labels_
    unique_labels = set(labels)
    os.makedirs('clusters', exist_ok=True)
    keypoints_file = os.path.join('clusters', 'all_keypoints.csv')
    if not os.path.exists(keypoints_file):
        with open(keypoints_file, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(["image_name", "x", "y"])
    with open(keypoints_file, 'a', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(sat_kpts)):
            wr.writerow([output_prefix, sat_kpts[i][0], sat_kpts[i][1]])
    cluster_dict = {}
    for lbl in unique_labels:
        mask = (labels == lbl)
        cluster_dict[lbl] = {
            'sat_kpts': sat_kpts[mask],
            'drone_kpts': drone_kpts[mask],
            'scores': scores[mask] if scores is not None else None
        }
    if plot and sat_image_original_color is not None and len(unique_labels) > 0:
        plt.imshow(sat_image_original_color)
        color_list = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for lbl, color in zip(unique_labels, color_list):
            mask = (labels == lbl)
            c_sat_kpts = sat_kpts[mask]
            plt.scatter(c_sat_kpts[:, 0], c_sat_kpts[:, 1], color=color, s=30, label=f'Cluster {lbl}')
        plt.title("All Clusters of Matches")
        plt.xlabel("X Coord")
        plt.ylabel("Y Coord")
        plt.legend()
        cluster_img_path = os.path.join('clusters', f'{output_prefix}_clusters.png')
        plt.savefig(cluster_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    return cluster_dict

def visualize_matches(image0, image1, feats0, feats1, matches, title='', save_path='matches', filename_prefix='', scale_factor=0.5):
    """
    Visualizes the matches between drone and satellite images.
    Expects image0 as the drone image tensor and image1 as the satellite image (BGR numpy array).
    """
    logging.info("Visualizing matches...")
    logging.info(f"Drone image shape: {image0.shape}")
    logging.info(f"Satellite image shape: {image1.shape}")
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matched_indices = matches.get("matches0", None)
    if matched_indices is None or len(matched_indices) == 0:
        logging.warning("No matches found to visualize.")
        return
    if isinstance(matched_indices, torch.Tensor):
        matched_indices = matched_indices.cpu().numpy()
    valid_mask = matched_indices > -1
    m_kpts0 = kpts0[valid_mask].cpu().numpy()
    m_kpts1 = kpts1[matched_indices[valid_mask]].cpu().numpy()
    n_matches = len(m_kpts0)
    logging.info(f"Number of matches to visualize: {n_matches}")
    if n_matches == 0:
        logging.warning("No valid matches after masking.")
        return
    if torch.is_tensor(image1):
        image1 = image1.cpu().permute(1, 2, 0).numpy()
        image1 = (image1 * 255).astype(np.uint8)
    h1, w1 = image1.shape[:2]
    new_w1 = int(w1 * scale_factor)
    new_h1 = int(h1 * scale_factor)
    image1_small = cv2.resize(image1, (new_w1, new_h1), interpolation=cv2.INTER_AREA)
    m_kpts1_small = m_kpts1 * scale_factor
    plt.figure(figsize=(12, 6))
    axes = viz2d.plot_images([image0, image1_small])
    viz2d.plot_matches(m_kpts0, m_kpts1_small, color="lime", lw=0.2)
    out_name = f'{filename_prefix}_{title}_matches.png'
    out_path = os.path.join(save_path, out_name)
    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved match visualization => {out_path}")
    except Exception as e:
        logging.error(f"Failed to save match visualization: {e}")
    finally:
        plt.close()

# -----------------------------------------------------
# SPATIAL INDEX & MATCHING OVER TILES
# -----------------------------------------------------

def build_spatial_index(features_list, tile_size=1000):
    """
    Builds an R-tree spatial index from the list of tile features.
    Each tile’s bounding box is defined by its (x, y) position and tile_size.
    """
    from rtree import index
    p = index.Property()
    p.dimension = 2
    idx = index.Index(properties=p)
    for i, tile in enumerate(features_list):
        x, y = tile['tile_position']
        idx.insert(i, (x, y, x + tile_size, y + tile_size))
    return idx

def get_overlapping_tiles_spatial_index(features_list, roi, spatial_index, tile_size=1000):
    """
    Retrieves tiles that overlap with the ROI using the spatial index.
    """
    matches = spatial_index.intersection(roi)
    overlapping_tiles = [features_list[i] for i in matches]
    logging.info(f"Found {len(overlapping_tiles)} overlapping tiles within ROI {roi} using spatial index.")
    return overlapping_tiles

def collect_and_sort_matches_overlapping_tiles(overlapping_tiles, matcher, feats_drone, device, sat_image_name):
    """
    For each overlapping tile, match its features with the drone features.
    Keypoint offsets are applied once here to convert local tile coordinates into global satellite coordinates.
    Returns a sorted list of match results.
    """
    all_m = []
    for i, tile in enumerate(tqdm(overlapping_tiles, desc=f"Matching overlapping tiles of {sat_image_name}")):
        feats_sat_local = tile["features"]
        x_tile, y_tile = tile["tile_position"]
        # Add tile offset once to obtain global coordinates
        global_keypoints = feats_sat_local["keypoints"] + torch.tensor([x_tile, y_tile], device=device)
        feats_sat = {
            'keypoints': global_keypoints,
            'descriptors': feats_sat_local['descriptors']
        }
        m = match_features(matcher, feats_drone, feats_sat, device)
        matched_idx = m.get("matches0", None)
        if matched_idx is not None:
            if isinstance(matched_idx, torch.Tensor):
                valid_mask = matched_idx > -1
                nv = valid_mask.sum().item()
            else:
                matched_idx = np.array(matched_idx)
                valid_mask = matched_idx > -1
                nv = int(valid_mask.sum())
            if nv > 0:
                all_m.append({'tile_idx': i, 'matches': m, 'num_valid': nv})
                logging.info(f"Tile {i} => {nv} matches.")
    all_m.sort(key=lambda x: x['num_valid'], reverse=True)
    return all_m

# -----------------------------------------------------
# OTHER HELPER FUNCTIONS
# -----------------------------------------------------

def latlon_to_xy(boundaries, sat_res, lat, lon):
    """
    Converts geographic coordinates (lat, lon) into pixel coordinates (x, y) for the satellite image.
    boundaries: [north, south, east, west]
    sat_res: [height, width]
    """
    north, south, east, west = boundaries
    if not (south <= lat <= north and west <= lon <= east):
        raise ValueError("Input latitude/longitude is outside the provided boundaries.")
    north_south = abs(north - south)
    east_west = abs(east - west)
    px_lon = east_west / sat_res[1]
    px_lat = north_south / sat_res[0]
    x = (lon - west) / px_lon
    y = (north - lat) / px_lat
    return x, y

def define_roi_around(x_center, y_center, width, height, roi_size=2000):
    """
    Defines an ROI (bounding box) centered at (x_center, y_center) with a given roi_size.
    Clamps the ROI to the image dimensions.
    """
    half = roi_size // 2
    x_min = int(max(0, x_center - half))
    y_min = int(max(0, y_center - half))
    x_max = int(min(width, x_center + half))
    y_max = int(min(height, y_center + half))
    return (x_min, y_min, x_max, y_max)

def verify_image_files(images_dir, image_names):
    missing_files = []
    for name in image_names:
        img_path = os.path.join(images_dir, name)
        if not os.path.isfile(img_path):
            missing_files.append(name)
    if missing_files:
        logging.error(f"Missing files in '{images_dir}': {missing_files}")
    else:
        logging.info(f"All files found in '{images_dir}'.")

def load_bgr_safe(img_path, size=None):
    """Loads an image in BGR format safely; returns a black image if loading fails."""
    if not img_path or not os.path.exists(img_path):
        if size is not None:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            return np.zeros((240, 320, 3), dtype=np.uint8)
    img = cv2.imread(img_path)
    if img is None:
        if size is not None:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            return np.zeros((240, 320, 3), dtype=np.uint8)
    if size is not None:
        img = cv2.resize(img, size)
    return img

def generate_drone_trajectory_plot(positions, boundaries, size=(640, 360)):
    """
    Generates a simple 2D plot of the drone trajectory over the given boundaries.
    Returns a BGR image.
    """
    if len(positions) < 1:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    north, south, east, west = boundaries
    w_img, h_img = size
    out = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    lon_min, lon_max = west, east
    lat_min, lat_max = south, north
    lon_range = max(1e-6, lon_max - lon_min)
    lat_range = max(1e-6, lat_max - lat_min)
    pts_px = []
    for (lon, lat) in positions:
        x_px = (lon - lon_min) / lon_range * w_img
        y_px = (lat_max - lat) / lat_range * h_img
        pts_px.append((int(x_px), int(y_px)))
    for i in range(len(pts_px)-1):
        cv2.line(out, pts_px[i], pts_px[i+1], (0, 255, 0), 2)
    if len(pts_px) > 0:
        cv2.circle(out, pts_px[-1], 5, (0, 0, 255), -1)
    cv2.putText(out, "Trajectory", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return out

# -----------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------

def main():
    """
    Main pipeline:
      - Set up devices and load images.
      - Rotate drone images based on yaw.
      - Extract features for drone images and satellite tiles (with caching).
      - Build a spatial index for the satellite tiles.
      - For each drone image: define an ROI, identify overlapping tiles, match features,
        perform clustering, estimate camera pose, and update last known position.
      - Save results and create a composite video.
    """
    STATISTICS = {}
    drone_images_dir = "vpair"
    satellite_images_dir = ""  # Set your satellite images directory
    drone_image_names = [f"{i:05}.png" for i in range(361, 370)]
    satellite_image_names = ["the_one.jpg"]

    boundaries_file = os.path.join("vpair", "SatData", "boundaries.txt")
    boundaries = load_bonuds(boundaries_file)
    yaw_csv_file = "poses.csv"
    yaw_angles = load_yaw_angles(yaw_csv_file)

    sat_image_path = os.path.join(satellite_images_dir, "the_one.jpg")
    sat_img = cv2.imread(sat_image_path, cv2.IMREAD_GRAYSCALE)
    if sat_img is None:
        logging.error(f"Failed to load satellite image from {sat_image_path}")
        return
    sat_res = [sat_img.shape[0], sat_img.shape[1]]

    camera_matrix = np.array([[750.62614972, 0.0, 402.41007535],
                              [0.0, 750.26301185, 292.98832147],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_coeffs = np.array([-0.11592226392258145, 0.1332261251415265, -0.00043977637330175616, 0.0002380609784102606], dtype=np.float32)
    pnp_solver = PnP(camera_matrix, dist_coeffs, ransac=False, ransac_iterations_count=10000, ransac_reprojection_error=2.0)
    last_known_position = [7.12153389, 50.74064904, 0]
    device = setup_device()
    extractor, matcher = initialize_models(device)

    # Load and process drone images
    drone_images = load_images(drone_images_dir, drone_image_names)
    logging.info(f"Loaded drone images: {list(drone_images.keys())}")
    for name, img_tensor in drone_images.items():
        if name in yaw_angles:
            yaw_deg = yaw_angles[name]
            rotated = rotate_tensor_image(img_tensor, -yaw_deg)
            drone_images[name] = rotated
            logging.info(f"Rotated {name} by {yaw_deg} deg.")
        else:
            logging.warning(f"No yaw for {name}. Skipping rotation.")

    # Load satellite images
    satellite_images = load_images(satellite_images_dir, satellite_image_names)

    # Extract or load drone features
    drone_feature_dir = "drone_features"
    if not os.path.exists(drone_feature_dir):
        os.makedirs(drone_feature_dir)
        drone_features = extract_features(extractor, drone_images, device)
        # Optionally, save features here.
    else:
        logging.info(f"Found precomputed drone features in '{drone_feature_dir}'.")
        # For simplicity, extract them here again.
        drone_features = extract_features(extractor, drone_images, device)

    # Process satellite image tiles
    satellite_tiles = {}
    satellite_features = {}
    spatial_indices = {}
    TILE_SIZE = 1500  # Adjust tile size as needed
    for sat_img_name, sat_img_tensor in satellite_images.items():
        tiles = split_image_into_tiles(sat_img_tensor, tile_size=(TILE_SIZE, TILE_SIZE), overlap=0, roi=None)
        visualize_tile_grid(tiles, sat_image_path, tile_size=(TILE_SIZE, TILE_SIZE))
        satellite_tiles[sat_img_name] = tiles
        features_tiles = extract_features_from_tiles_and_cache(tiles, extractor, device, sat_img_name)
        satellite_features[sat_img_name] = features_tiles
        spatial_index = build_spatial_index(features_tiles, tile_size=TILE_SIZE)
        spatial_indices[sat_img_name] = spatial_index

    os.makedirs('camera_positions', exist_ok=True)
    camera_positions_csv = os.path.join('camera_positions', 'camera_positions.csv')
    if not os.path.exists(camera_positions_csv):
        with open(camera_positions_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["drone_image", "longitude", "latitude"])

    video_frames_data = []
    positions_so_far = []

    # Process each drone image
    for drone_img_name, feats_drn in drone_features.items():
        Start_time = time.time()
        logging.info(f"\n--- Processing {drone_img_name} ---")
        drone_img_tensor = drone_images[drone_img_name]
        lon, lat, _ = last_known_position
        x_center, y_center = latlon_to_xy(boundaries, sat_res, lat, lon)
        logging.info(f"Last known position (in pixels): ({x_center}, {y_center})")
        ROI_SIZE = 20000  # Adjust as needed
        roi_box = define_roi_around(x_center, y_center, width=sat_res[1], height=sat_res[0], roi_size=ROI_SIZE)
        x_min, y_min, x_max, y_max = roi_box

        sat_img_name = "the_one.jpg"  # Assuming single satellite image
        features_tiles = satellite_features[sat_img_name]
        spatial_index = spatial_indices[sat_img_name]
        overlapping_tiles = get_overlapping_tiles_spatial_index(features_tiles, roi_box, spatial_index, tile_size=TILE_SIZE)
        logging.info(f"Found {len(overlapping_tiles)} overlapping tiles before filtering.")
        min_keypoints_required = 10
        filtered_tiles = [tile for tile in overlapping_tiles if tile["features"]["keypoints"].shape[0] >= min_keypoints_required]
        logging.info(f"Filtered overlapping tiles: {len(filtered_tiles)} pass the threshold of {min_keypoints_required} keypoints.")
        overlapping_tiles = filtered_tiles

        if not overlapping_tiles:
            logging.warning(f"No overlapping tiles found for {drone_img_name}. Skipping matching.")
            best_cam = last_known_position.copy()
            best_match_path = None
            cluster_img_path = None
        else:
            all_matches = collect_and_sort_matches_overlapping_tiles(overlapping_tiles, matcher, feats_drn, device, sat_img_name)
            aggregated_sat_kpts = []
            aggregated_drone_kpts = []
            if len(all_matches) > 0:
                best_match = all_matches[0]
                best_tile_idx = best_match["tile_idx"]
                best_matches = best_match["matches"]
                sat_img_color = cv2.imread(sat_image_path)
                visualize_matches(
                    image0=drone_img_tensor,         # Drone image tensor
                    image1=sat_img_color,              # Satellite image (BGR)
                    feats0=drone_features[drone_img_name],
                    feats1=overlapping_tiles[best_tile_idx]['features'],
                    matches=best_matches,
                    title=f"{drone_img_name}_Tile_{best_tile_idx}",
                    save_path='matches',
                    filename_prefix=f"{drone_img_name}_{sat_img_name}_Tile_{best_tile_idx}",
                    scale_factor=0.5
                )
            for match in all_matches:
                m = match['matches']
                matched_idx = m.get("matches0", None)
                if matched_idx is None:
                    continue
                if isinstance(matched_idx, torch.Tensor):
                    valid_mask = matched_idx > -1
                    m_kpts0 = feats_drn["keypoints"][valid_mask].cpu().numpy()
                else:
                    matched_idx = np.array(matched_idx)
                    valid_mask = matched_idx > -1
                    m_kpts0 = feats_drn["keypoints"][valid_mask].cpu().numpy()
                tile_idx = match['tile_idx']
                tile = overlapping_tiles[tile_idx]
                x_tile, y_tile = tile['tile_position']
                feats_tile = tile['features']
                max_idx = feats_tile['keypoints'].shape[0]
                matched_idx_clipped = np.clip(matched_idx[valid_mask], 0, max_idx - 1)
                m_kpts1 = feats_tile['keypoints'][matched_idx_clipped].cpu().numpy() + np.array([x_tile, y_tile])
                aggregated_sat_kpts.append(m_kpts1)
                aggregated_drone_kpts.append(m_kpts0)
            if len(aggregated_sat_kpts) > 0 and len(aggregated_drone_kpts) > 0:
                all_sat_kpts = np.vstack(aggregated_sat_kpts)
                all_drone_kpts = np.vstack(aggregated_drone_kpts)
                cluster_dict = visualize_clusters_final(
                    all_sat_kpts, all_drone_kpts, None,
                    sat_image_path,
                    plot=True,
                    output_prefix=drone_img_name.replace('.', '_')
                )
                cluster_img_path = os.path.join('clusters', f'{drone_img_name.replace(".", "_")}_clusters.png')
                clusters = defaultdict(list)
                for lbl, dct in cluster_dict.items():
                    s_kpts = dct['sat_kpts']
                    d_kpts = dct['drone_kpts']
                    for sp, dp in zip(s_kpts, d_kpts):
                        clusters[lbl].append({'sat_pt': sp, 'drone_pt': dp, 'score': None})
                cluster_conf = estimate_camera_pose(clusters, boundaries, sat_res, pnp_solver, last_known_position)
                best_lbl, best_cam = select_best_cluster(cluster_conf, boundaries, last_known_position)
                if best_cam is not None:
                    last_known_position = [best_cam[0], best_cam[1], 0.0]
                    logging.info(f"Selected cluster => {best_lbl}, cam pos => {best_cam}")
                    g_time = time.time()
                    logging.info(f"Time taken for {drone_img_name}: {g_time - Start_time:.2f} sec")
                    best_match_path = None
                else:
                    logging.warning("No valid cluster. Retaining last known position.")
                    best_cam = last_known_position.copy()
                    best_match_path = None
                    cluster_img_path = None
                with open(camera_positions_csv, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([drone_img_name, best_cam[0], best_cam[1]])
            else:
                logging.warning(f"No valid matches for {drone_img_name}; skipping PnP.")
                best_cam = last_known_position.copy()
                best_match_path = None
                cluster_img_path = None

        drone_img_path = os.path.join(drone_images_dir, drone_img_name)
        positions_so_far.append((best_cam[0], best_cam[1]))
        video_frames_data.append({
            "drone_img_name":  drone_img_name,
            "drone_img_path":  drone_img_path,
            "best_match_path": best_match_path,
            "cluster_img_path": cluster_img_path,
            "positions_up_to_now": positions_so_far.copy(),
        })

        ground_truth = {}
        try:
            with open('poses.csv', 'r') as f:
                rd = csv.DictReader(f)
                for row in rd:
                    ground_truth[row['filename']] = {
                        'lat': float(row.get('lat', 0.0)),
                        'lon': float(row.get('lon', 0.0))
                    }
        except:
            logging.warning("No valid ground truth info found in poses.csv")
        if drone_img_name in ground_truth:
            gt = ground_truth[drone_img_name]
            def haversine_distance(lat1, lon1, lat2, lon2):
                R = 6371000
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                return R * c
            err_m = haversine_distance(best_cam[1], best_cam[0], gt['lat'], gt['lon'])
            logging.info(f"Ground truth error: {err_m:.2f} m")
            STATISTICS[drone_img_name] = {
                'predicted_lat': best_cam[1],
                'predicted_lon': best_cam[0],
                'ground_truth_lat': gt['lat'],
                'ground_truth_lon': gt['lon'],
                'error_meters': err_m
            }
        else:
            STATISTICS[drone_img_name] = {
                'predicted_lat': best_cam[1],
                'predicted_lon': best_cam[0]
            }

    with open('statistics_endelig.csv', 'w', newline='') as f:
        fieldnames = ['filename','predicted_lat','predicted_lon','ground_truth_lat','ground_truth_lon','error_meters']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fn, d in STATISTICS.items():
            row = {'filename': fn,
                   'predicted_lat': d.get('predicted_lat', 0.0),
                   'predicted_lon': d.get('predicted_lon', 0.0),
                   'ground_truth_lat': d.get('ground_truth_lat', 0.0),
                   'ground_truth_lon': d.get('ground_truth_lon', 0.0),
                   'error_meters': d.get('error_meters', -1.0)}
            w.writerow(row)
    logging.info("Pipeline complete. Stats saved to statistics_endelig.csv.")
    logging.info("Generating composite video...")
    create_composite_video(video_frames_data, boundaries, output_path="final_composite_video.mp4", fps=2)

# -----------------------------------------------------
# COMPOSITE VIDEO CREATION FUNCTIONS
# -----------------------------------------------------

def create_composite_video(frames_data, boundaries, output_path, fps=5):
    """
    Creates a composite video that combines match visualization, cluster visualization,
    and the drone trajectory.
    """
    if frames_data[0]["best_match_path"] and os.path.exists(frames_data[0]["best_match_path"]):
        match_img = cv2.imread(frames_data[0]["best_match_path"])
        if match_img is not None:
            max_width = 1920
            match_h, match_w = match_img.shape[:2]
            match_aspect = match_w / match_h
            frame_w = min(max_width, match_w)
            match_target_h = int(frame_w / match_aspect)
            first_cluster = frames_data[0]["cluster_img_path"]
            if first_cluster and os.path.exists(first_cluster):
                cluster_img = cv2.imread(first_cluster)
                if cluster_img is not None:
                    cluster_h, cluster_w = cluster_img.shape[:2]
                    cluster_aspect = cluster_w / cluster_h
                    bottom_h = int(frame_w/2 / cluster_aspect)
                    frame_size = (frame_w, match_target_h + bottom_h)
                else:
                    frame_size = (1920, 1080)
                    frame_w, match_target_h, bottom_h = 1920, 720, 360
            else:
                frame_size = (1920, 1080)
                frame_w, match_target_h, bottom_h = 1920, 720, 360
    else:
        frame_size = (1920, 1080)
        frame_w, match_target_h, bottom_h = 1920, 720, 360
    logging.info(f"Video frame size: {frame_size}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for idx, frame_info in enumerate(frames_data):
        match_bgr = load_bgr_safe(frame_info["best_match_path"])
        if match_bgr is not None:
            match_bgr = cv2.resize(match_bgr, (frame_w, match_target_h))
        else:
            match_bgr = np.zeros((match_target_h, frame_w, 3), dtype=np.uint8)
        cluster_bgr = load_bgr_safe(frame_info["cluster_img_path"])
        if cluster_bgr is not None:
            cluster_bgr = cv2.resize(cluster_bgr, (frame_w//2, bottom_h))
        else:
            cluster_bgr = np.zeros((bottom_h, frame_w//2, 3), dtype=np.uint8)
        traj_bgr = generate_drone_trajectory_plot(frame_info["positions_up_to_now"], boundaries, size=(frame_w//2, bottom_h))
        bottom_half = np.hstack((cluster_bgr, traj_bgr))
        composite = np.vstack((match_bgr, bottom_half))
        writer.write(composite)
        logging.info(f"[CompositeVideo] Frame {idx+1}/{len(frames_data)} added.")
    writer.release()
    logging.info(f"[CompositeVideo] Saved => {output_path}")

# -----------------------------------------------------
# SCRIPT EXECUTION
# -----------------------------------------------------

if __name__ == "__main__":
    main()
