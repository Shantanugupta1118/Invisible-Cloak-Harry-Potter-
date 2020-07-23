import numpy as np
import cv2


def detect_blue(frame, background):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Create HSV color mask and segment the image based on Blue color
    sensitivity = 20
    H_Value = 20    # Change this value if you want to segment some other color
    light_blue = np.array([H_Value - sensitivity, 60, 60])
    dark_blue = np.array([H_Value + sensitivity, 255, 255])
    # Creating a segmentaion mask for the blue color
    mask = cv2.inRange(hsv_image, light_blue, dark_blue)

    # Apply closing operation to fill out the unwanted gaps in the image. Bigger the kernel size, lesser the gaps
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    print(kernel.mean(),"  ",kernel.shape,"  ",H_Value," ",light_blue,"  ",dark_blue)
    # Find the contour coordinates of the biggest area and create a mask of that area
    contours, _ = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    contour_mask = cv2.fillPoly(np.zeros((720,720, 3), dtype=np.uint8), pts=[
                                cont_sorted[0]], color=(255, 255, 255))

    # create the two masks with the background image and the main object mask such that we can superimpose them together
    object_mask = cv2.fillPoly(frame, pts=[cont_sorted[0]], color=(0, 0, 0))
    background_mask = np.bitwise_and(contour_mask, background)

    print(background.shape,"  ",background_mask.mean(),"  ",background_mask.mean())

    # Final image is created by doing a bitwise and of the two masks, which in turn removes the color in question
    # and replaces it with the background
    final_img = cv2.bitwise_or(object_mask, background_mask)

    return final_img


# Initiate video capture from source '0.
# Change the source value if you have more than one webcam and wish to use the secondary one
cap = cv2.VideoCapture(1)

# Read the background initially, resize it and then show it, wait for the user to press a key before continuing
ret, background = cap.read()
background = cv2.resize(background, (720,720))
cv2.imshow('Background', background)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the video in MP4 format in the same directory as the code (Optional)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (720,720))

while(True):
    # Capture frame-by-frame, resize and then apply the invisibility algorithm
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720,720))
    image = detect_blue(frame, background)

    # The processed frame is added to the video
    out.write(image)

    # Display the resulting video
    cv2.imshow('Image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all assets
cap.release()
out.release()
cv2.destroyAllWindows()
