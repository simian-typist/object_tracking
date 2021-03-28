import cv2

import argparse as ap
import numpy as np

from time import time
from yolov4.tf import YOLOv4


def read_video(vidname):
    cap = cv2.VideoCapture(vidname)

    # initialise detector
    # TODO parameterise detector
    # Make class -> have its own config?
    yolo = YOLOv4()

    yolo.config.parse_names("yolo/coco.names")
    yolo.config.parse_cfg("yolo/csp.cfg")

    yolo.make_model()
    yolo.load_weights("yolo/yolov4-csp.weights", weights_type="yolo")
    yolo.summary(summary_type="yolo")
    yolo.summary()

    cv2.namedWindow("greg")
    ok, frame = cap.read()
    while ok:
        # TODO read size from config

        start = time()
        bboxes = yolo.predict(frame, 0.3)
        print("inference took: {}".format(time() - start))
        # TODO np.where is faster
        bboxes = bboxes[bboxes[:, 4] == 0]  # only look at people

        frame = yolo.draw_bboxes(frame, bboxes)
        cv2.imshow("greg", frame)
        ch = cv2.waitKey(1) & 0xFF
        if ch == ord("q"):
            break
        ok, frame = cap.read()
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        default="/home/torch/Downloads/motvid.webm",
        help="Path to input file",
    )
    args = parser.parse_args()
    read_video(args.input_file)
