import os
import time
import math
import cv2
import numpy as np
import depthai as dai
import json

def save_img_and_box(name, img, data):
    folder = "out/"
    cv2.imwrite(f"{folder}/{name}.png", img)
    with open(f"{folder}/{name}.json", 'w') as fp:
        json.dump(data, fp)

class OAKTrapPipeline:
    def __init__(self, nnBlobPath, labelMap, is_online=True):
        nnBlobPath = os.path.abspath(nnBlobPath)

        if not os.path.exists(nnBlobPath):
            raise FileNotFoundError(f'Required file/s not found: "{nnBlobPath}"')

        self.labelMap = labelMap
        self.is_online = is_online

        # oak_aspect_ratio = 4056 / 3040
        self.res_rgb_wh = [416, 416]

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()

        # Define a source - color camera
        self.detectionNetwork = self.pipeline.create(dai.node.YoloDetectionNetwork)

        if self.is_online:
            self.rgbCam = self.pipeline.createColorCamera()
            self.rgbCam.setPreviewSize(self.res_rgb_wh[0], self.res_rgb_wh[1])
            self.rgbCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            self.rgbCam.setInterleaved(False)
            self.rgbCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        else:
            # use video file or youtube link
            self.imgInput = RGBCamNode_online()
            self.imgInput.setStreamName("imgInput")
            self.rgbCam = self.pipeline.create(dai.node.ImageManip)

        # create output nodes
        self.xoutRgb = self.pipeline.createXLinkOut()
        # self.xoutRgbHD = self.pipeline.createXLinkOut()
        self.xoutNN = self.pipeline.createXLinkOut()

        self.xoutRgb.setStreamName("rgb")
        # self.xoutRgbHD.setStreamName("rgbHD")
        self.xoutNN.setStreamName("detections")

        # manip = self.pipeline.createImageManip()
        # manip.setMaxOutputFrameSize(1280 * 720 * 3)
        # manip.initialConfig.setResizeThumbnail(int(1280 * oak_aspect_ratio), 1280)

        self.detectionNetwork.setBlobPath(nnBlobPath)
        self.detectionNetwork.setConfidenceThreshold(0.5)
        self.detectionNetwork.setNumClasses(80)
        self.detectionNetwork.setCoordinateSize(4)
        self.detectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
        self.detectionNetwork.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
        self.detectionNetwork.setIouThreshold(0.5)

        # Connect nodes
        if self.is_online:
            self.rgbCam.preview.link(self.detectionNetwork.input)
            self.rgbCam.preview.link(self.xoutRgb.input)
            # self.rgbCam.video.link(self.xoutRgbHD.input)
        else:
            self.rgbCam.out.link(self.detectionNetwork.input)
            self.rgbCam.out.link(self.xoutRgb.input)

        self.detectionNetwork.out.link(self.xoutNN.input)


    def start(self):
        # Pipeline is defined, now we can connect to the device
        with dai.Device(self.pipeline) as device:
            # Start pipeline
            device.startPipeline()

            if not self.is_online:
                oakd_play = rec.OAKDPlayer(device, "./recorded/2022.02.06 13.17.13")

            # Output queues will be used to get frames from OAK-D camera
            rgbPreviewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            # rgbVideoQueue = device.getOutputQueue(name="rgbHD", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

            camFrame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)

            is_first_pass = True
            while True:
                if not self.is_online:
                    oakd_play.sendNextFrameToOAKD()


                inRgbPreview = rgbPreviewQueue.get()
                # inRgbVideo = rgbPreviewQueue.get()
                inNN = detectionNNQueue.get()

                counter += 1
                currentTime = time.monotonic()
                if (currentTime - startTime) > 1:
                    fps = counter / (currentTime - startTime)
                    counter = 0
                    startTime = currentTime

                camFrame = inRgbPreview.getCvFrame()
                # videoFrame = inRgbVideo.getCvFrame()
                detections = inNN.detections
                print(detections)
                if not self.is_online:
                    camFrame.shape += (1, 1)
                    camFrame = np.reshape(camFrame, newshape=(3, self.res_rgb_wh[1], self.res_rgb_wh[0])).transpose(1, 2, 0).copy()


                if len(detections) != 0:
                    cv2.imwrite("out/img.png", camFrame)
                    if self.is_online:
                        detections = [d for d in detections if d.label in self.labelMap.keys()]
                        if len(detections) != 0:
                            data = []
                            H, W = camFrame.shape[:2]
                            for d in detections:
                                data.append({
                                    "left_top_xy": [int(d.xmin*W), int(d.ymin*H)],
                                    "right_bottom_xy": [int(d.xmax*W), int(d.ymax*H)],
                                    "label": d.label,
                                    "label_str": self.labelMap[d.label],
                                    "confidence": d.confidence,
                                })
                            save_img_and_box("test", camFrame, data)
                    # else:
                    #     boundingBoxMapping = xoutBoundingBoxMapping.get()
                    #     roiDatas = boundingBoxMapping.getConfigData()
                    #     for i in range(len(detections)):
                    #         detection = detections[i]
                    #         roiDatas_i = roiDatas[i:i+1]
                    #         if detection.label not in self.labelMap.keys():
                    #             continue
                    #         boxes += extract_boxes(roiDatas_i)

                    # draw centroids on camera frame
                    # for ctd in boxes:
                    #     cv2.rectangle(camFrame, ctd["box_topleft_xy"], ctd["box_bottomright_xy"], (0, 255, 0), 2)



                cv2.putText(camFrame, "NN fps: {:.2f}".format(fps), (2, camFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                # display images
                cv2.imshow("OAK-D Preview", camFrame)
                if cv2.waitKey(1) == ord('q'):
                    break

def main():
    # Tiny yolo v3/4 label texts
    labelMap = {
        0: "person",
        # 14: "bird",
        # 15: "cat",
        16: "dog",
        # 17: "horse",
        # 18: "sheep",
        # 19: "cow",
        # 20: "elephant",
        # 21: "bear",
        # 22: "zebra",
        # 23: "giraffe",
    }

    epl = OAKTrapPipeline(
        nnBlobPath='models/tiny-yolo-v4_openvino_2021.2_6shave.blob',
        labelMap=labelMap,
        is_online=True,
    )
    epl.start()

if __name__ == '__main__':
    main()
"""
[
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

0 person
1 bicycle
2 car
3 motorbike
4 aeroplane
5 bus
6 train
7 truck
8 boat
9 traffic light
10 fire hydrant
11 stop sign
12 parking meter
13 bench
14 bird
15 cat
16 dog
17 horse
18 sheep
19 cow
20 elephant
21 bear
22 zebra
23 giraffe
24 backpack
25 umbrella
26 handbag
27 tie
28 suitcase
29 frisbee
30 skis
31 snowboard
32 sports ball
33 kite
34 baseball bat
35 baseball glove
36 skateboard
37 surfboard
38 tennis racket
39 bottle
40 wine glass
41 cup
42 fork
43 knife
44 spoon
45 bowl
46 banana
47 apple
48 sandwich
49 orange
50 broccoli
51 carrot
52 hot dog
53 pizza
54 donut
55 cake
56 chair
57 sofa
58 pottedplant
59 bed
60 diningtable
61 toilet
62 tvmonitor
63 laptop
64 mouse
65 remote
66 keyboard
67 cell phone
68 microwave
69 oven
70 toaster
71 sink
72 refrigerator
73 book
74 clock
75 vase
76 scissors
77 teddy bear
78 hair drier
79 toothbrush


0 person
14 bird
15 cat
16 dog
17 horse
18 sheep
19 cow
20 elephant
21 bear
22 zebra
23 giraffe
"""
