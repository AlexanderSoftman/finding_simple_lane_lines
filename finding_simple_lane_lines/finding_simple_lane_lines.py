import logging
import finding_simple_lane_lines
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


os_path_test_images = "/home/afomin/projects/mj/finding_simple_lane_lines/finding_simple_lane_lines/test_images/"
os_path_test_videos = "/home/afomin/projects/mj/finding_simple_lane_lines/finding_simple_lane_lines/test_videos/"

test_images = (
    ("solidWhiteCurve.jpg", True),
    ("solidWhiteRight.jpg", True),
    ("solidYellowCurve.jpg", True),
    ("solidYellowCurve2.jpg", True),
    ("solidYellowLeft.jpg", True),
    ("whiteCarLaneSwitch.jpg", True))

test_videos = (
    ("challenge.mp4", True),
    ("solidWhiteRight.mp4", False),
    ("solidYellowLeft.mp4", False))


# The tools you have are color selection, 
# region of interest selection, 
# grayscaling, 
# Gaussian smoothing, 
# Canny Edge Detection and 
# Hough Tranform line detection.

def analyze(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    frame_res = np.copy(frame)
    # convert frame to gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # add gaussian smoothing to gray frame
    kernel_size = 5
    blur_gray_frame = cv2.GaussianBlur(
        gray_frame,
        (kernel_size, kernel_size),
        0)

    # define Canny parameters:
    low_threshold = 50
    high_threshold = 150
    canny_edges_from_gray_frame = cv2.Canny(
        blur_gray_frame,
        low_threshold,
        high_threshold)

    # define Hough transformation parameters
    rho = 1
    theta = np.pi / 180
    threshold = 1
    min_line_length = 15
    max_line_gap = 15

    # create empty image
    line_frame = np.copy(frame) * 0
    lines = cv2.HoughLinesP(
        canny_edges_from_gray_frame,
        rho,
        theta,
        threshold,
        np.array([]),
        min_line_length,
        max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(
                line_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                10)

    # add region selections
    polygon_dots = [[
        [width * 80 / 960, height * 535 / 540],
        [width * 402 / 960, height * 346 / 540],
        [width * 576 / 960, height * 346 / 540],
        [width * 916 / 960, height * 535 / 540]]]

    frame_filter = np.zeros(
        [
            frame.shape[0],
            frame.shape[1],
            frame.shape[2]
        ],
        dtype=np.uint8)

    window = np.array(polygon_dots, dtype=np.int32)

    cv2.fillPoly(
        frame_filter,
        window,
        [1, 1, 1])

    lines_filtered = frame_filter * line_frame
    # add lines to color_edges
    frame_res = cv2.addWeighted(
        frame,
        0.8,
        lines_filtered,
        1,
        0)
    return(frame_res)


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s')
    # for test_image in test_images:
    #     if test_image[1] is True:
    #         image = mpimg.imread(os_path_test_images + test_image[0])
    #         plt.imshow(analyze(image))
    #         plt.show()

    for test_video in test_videos:
        if test_video[1] is True:
            print(os_path_test_images + test_video[0])
            cap = cv2.VideoCapture(os_path_test_videos + test_video[0])
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                # Display the resulting frame
                cv2.imshow(
                    test_video[0],
                    cv2.cvtColor(
                        analyze(
                            cv2.cvtColor(
                                frame,
                                cv2.COLOR_BGR2RGB)),
                        cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
