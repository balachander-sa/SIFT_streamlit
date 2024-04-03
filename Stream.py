import streamlit as st
import cv2
import numpy as np
import time

def sift_feature_matching(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT feature extraction
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3

def main():
    st.title("Live Video Stream with SIFT Feature Matching by Balachander")

    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to open camera.")
        return

    # Initialize variables
    prev_frame = None
    processed_frame = None

    # Main loop
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            break

        # Perform SIFT feature matching if there's a previous frame
        if prev_frame is not None:
            processed_frame = sift_feature_matching(prev_frame, frame)
        else:
            processed_frame = frame

        # Display processed frame with SIFT feature matching
        st.image(processed_frame, channels="BGR", use_column_width=True, caption='SIFT Feature Matching')

        # Update previous frame
        prev_frame = frame.copy()

        # Wait for a short duration
        time.sleep(0.1)

    # Release video capture
    cap.release()

if __name__ == "__main__":
    main()
