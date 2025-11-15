"""
Interactive HSV color tuning tool for grain masking.

This tool provides an interactive GUI with trackbars to help determine
the optimal HSV color range for masking rice grains under different
lighting conditions.
"""

import cv2
import numpy as np


def nothing(x):
    """Callback function for trackbar (does nothing)."""
    pass


def tune_hsv_values(image_path, initial_values=None):
    """
    Launch interactive HSV tuning tool.
    
    Args:
        image_path (str): Path to the image to tune
        initial_values (dict, optional): Initial HSV values. Default values:
            {'H': 70, 'S': 110, 'V': 0, 'H2': 109, 'S2': 255, 'V2': 255}
    
    Usage:
        - Adjust trackbars to find optimal HSV range
        - Press 'q' to quit
        - The tool shows the masked result in real-time
    """
    if initial_values is None:
        initial_values = {
            'H': 70, 'S': 110, 'V': 0,
            'H2': 109, 'S2': 255, 'V2': 255
        }
    
    window_name = "HSV Tuner - Press 'q' to quit"
    
    # Create window and trackbars
    cv2.namedWindow(window_name)
    cv2.createTrackbar('H_min', window_name, initial_values['H'], 179, nothing)
    cv2.createTrackbar('S_min', window_name, initial_values['S'], 255, nothing)
    cv2.createTrackbar('V_min', window_name, initial_values['V'], 255, nothing)
    cv2.createTrackbar('H_max', window_name, initial_values['H2'], 179, nothing)
    cv2.createTrackbar('S_max', window_name, initial_values['S2'], 255, nothing)
    cv2.createTrackbar('V_max', window_name, initial_values['V2'], 255, nothing)
    
    # Load and resize image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Resize for display
    scale_percent = 30
    width = int(img.shape[1] * scale_percent / 400)
    height = int(img.shape[0] * scale_percent / 400)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    print(f"\nHSV Tuner launched for: {image_path}")
    print("Adjust the trackbars to find optimal HSV range")
    print("Press 'q' to quit\n")
    
    while True:
        # Get current trackbar positions
        h_min = cv2.getTrackbarPos('H_min', window_name)
        s_min = cv2.getTrackbarPos('S_min', window_name)
        v_min = cv2.getTrackbarPos('V_min', window_name)
        h_max = cv2.getTrackbarPos('H_max', window_name)
        s_max = cv2.getTrackbarPos('S_max', window_name)
        v_max = cv2.getTrackbarPos('V_max', window_name)
        
        # Convert to HSV and apply mask
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        lower_boundary = np.array([h_min, s_min, v_min])
        upper_boundary = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_boundary, upper_boundary)
        
        # Apply mask (invert to keep grain, remove background)
        final = cv2.bitwise_and(img_resized, img_resized, mask=cv2.bitwise_not(mask))
        
        # Display result
        cv2.imshow(window_name, final)
        
        # Print current values
        key = cv2.waitKey(1)
        if key == ord('q'):
            print(f"\nFinal HSV Values:")
            print(f"Lower bound: H={h_min}, S={s_min}, V={v_min}")
            print(f"Upper bound: H={h_max}, S={s_max}, V={v_max}")
            print(f"\nUse these values in masking.py:")
            print(f"lower = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper = np.array([{h_max}, {s_max}, {v_max}])")
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    img_path = r"../../data/images/DSC01902.JPG"
    tune_hsv_values(img_path)

