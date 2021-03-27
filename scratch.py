import argparse as ap
import cv2
import numpy as np


def read_video(vidname):
    cap = cv2.VideoCapture(vidname)

    # initialise detector
    # TODO parameterise detector
    # Make class -> have its own config?
    net = cv2.dnn.readNetFromDarknet("yolo/config", "yolo/yolov3.weights")

    cv2.namedWindow("greg")
    ok, frame = cap.read()
    while ok:
        # TODO read size from config
        img_blb = cv2.dnn.blobFromImage(
            frame, scalefactor=(1 / 255.0), size=(608, 608), swapRB=True, crop=False
        )

        net.setInput(img_blb)
        ret = net.forward()
        boxes = []
        confidences = []
        classIDs = []
        for detection in ret:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.3:
                width = detection[2]
                height = detection[3]
                top_x = frame.shape[1] * (detection[0] - 0.5 * width)
                top_y = frame.shape[0] * (detection[1] - 0.5 * height)
                width = int(frame.shape[1] * (detection[2]))
                height = int(frame.shape[0] * (detection[3]))
                boxes.append([top_x, top_y, width, height])
                confidences.append(float(confidence))
                classIDs.append(classID)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        print(len(indices))
        if len(indices) > 0:
            for idx in indices.flatten():
                x = int(boxes[idx][0])
                y = int(boxes[idx][1])
                w = boxes[idx][2]
                h = boxes[idx][3]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))

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
