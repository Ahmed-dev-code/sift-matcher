# Image Matching with SIFT 

This project demonstrates image matching using **Scale-Invariant Feature Transform (SIFT)** and **Brute-Force Matching (BFMatcher)**. The goal is to identify the best matching image in a training set for a given query image, based on feature detection.

## Features
- Detect and display keypoints in an image
- Find matches between a query image and training images
- Display the best matching image with highlighted matches

---

## Setup

### Prerequisites
- Python 3.x
- Required packages: `numpy`, `opencv-python-headless`, and `matplotlib`

### Installation
Install the dependencies by running:
```bash
pip install numpy opencv-python-headless matplotlib
```

## Folder structure
```bash
dataset/
├── test/
│   └── tomato.jpg           # Query image
└── training/
    ├── fruits&veg1.jpg      # Training images
    ├── fruits&veg2.jpg
    ├── fruits&veg3.jpg
    ├── fruits&veg4.jpg
    └── fruits&veg5.jpg
```


## Usage

1. Place your query image in `dataset/test/`. (Update the `image_query_path` in the main code)
2. Add training images in `dataset/training/` with filenames like `fruits&veg1.jpg`, `fruits&veg2.jpg`, etc.
3. Run the script to identify the best match.

## Code Explanation

### Functions

- **`find_matches_between_two_images(img1_path, img2_path)`**
  - **Input:** Paths to two images
  - **Output:** Keypoints and matches between the two images
  - **Description:** Detects keypoints using SIFT and finds matches using BFMatcher. Matches are sorted based on distance, with lower distances indicating better matches.

- **`show_matches(img1_path, img2_path, kp1, kp2, matches)`**
  - **Input:** Paths to two images, keypoints, and matches
  - **Output:** Displays matching features between images
  - **Description:** Converts images to RGB and uses `cv2.drawMatches` to visualize matched features between the query and the best matching training image.

- **`draw_keypoints(img_path)`**
  - **Input:** Path to an image
  - **Output:** Displays the image with detected keypoints
  - **Description:** Draws keypoints on the image using SIFT and displays it with Matplotlib.

### Script Logic

1. **Draw Keypoints on Query Image:** The `draw_keypoints` function is used to visualize keypoints in the query image.
2. **Find the Best Match:**
   - Loop through all training images in the `dataset/training` directory.
   - For each image, find matches with the query image.
   - Track the image with the highest number of matches.
3. **Display Best Match:** The `show_matches` function highlights the matching features between the query and the best matching training image.
