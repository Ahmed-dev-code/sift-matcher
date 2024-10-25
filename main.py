import numpy as np
import cv2
from matplotlib import pyplot as plt

def find_matches_between_two_images(img1_path, img2_path):
    """Given two images, returns the matches and keypoints"""
    # Initiate SIFT detector use threshold 
    sift = cv2.SIFT_create()
    
    img1 = cv2.imread(img1_path)  # queryImage
    img2 = cv2.imread(img2_path)  # trainImage

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # sort the matches based on distance
    matches = sorted(matches, key=lambda val: val.distance)

    return kp1, kp2, matches

def show_matches(img1_path, img2_path, kp1, kp2, matches):
    """Shows the matches between two images"""
    img1 = cv2.imread(img1_path)  # queryImage
    img2 = cv2.imread(img2_path)  # trainImage

    # # Convert BGR to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img3 = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, matches, None, flags=2)
    plt.imshow(img3), plt.show()
    
def draw_keypoints(img_path) :
    """Draws the keypoints on the image"""
    img = cv2.imread(img_path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img),plt.show()

query_img_path = 'dataset/test/tomato.jpg'
draw_keypoints(query_img_path)

best_match = None
best_match_img_path = ''

for i in range(1, 6): # loop through the 5 images in the training set
    img_path = 'dataset/training/fruits&veg' + str(i) + '.jpg'
    print('Comparing', query_img_path, 'with', img_path)
    kp1, kp2, matches = find_matches_between_two_images(query_img_path, img_path)
    
    if best_match is None or len(matches) > len(best_match):
        best_match = matches
        best_match_img_path = img_path

# Display the best match
if best_match is not None:
    print('Best match:', best_match_img_path)
    show_matches(query_img_path, best_match_img_path, *find_matches_between_two_images(query_img_path, best_match_img_path))

