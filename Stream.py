import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import cv2
import numpy as np
import av

class SIFTFeatureMatchingTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_frame = None

    def sift_feature_matching(self, img1, img2):
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

    def recv(self, frame):
        if frame is None:
            return

        # Convert frame to numpy array
        frame = frame.to_ndarray(format="bgr24")

        # Perform SIFT feature matching if there's a previous frame
        if self.prev_frame is not None:
            processed_frame = self.sift_feature_matching(self.prev_frame, frame)
        else:
            processed_frame = frame

        # Update previous frame
        self.prev_frame = frame.copy()

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def main():
    st.title("Live Video Stream with SIFT Feature Matching")

    # Set client settings for WebRTC
    client_settings = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    # Render the video stream with SIFT feature matching
    webrtc_streamer(
        key="example",
        video_transformer_factory=SIFTFeatureMatchingTransformer,
        client_settings=client_settings,
    )

if __name__ == "__main__":
    main()
