import cv2
import numpy as np
import depthai as dai
import pdb
import time
from FPS import FPS, now


class RgbdReader():
    def __init__(self,internal_fps=35,downscaleColor=True):
        '''
        downscaleColor:If set (True), the ColorCamera is downscaled from 1080p to 720p
                    Otherwise (False), the aligned depth is automatically upscaled to 1080p
        internal_fps: fps for internal camera
        '''
        self.downscaleColor = downscaleColor
        self.internal_fps = internal_fps
        self.device = dai.Device()
        self.device.startPipeline(self.create_pipeline())
        self.fps = FPS()
    
    def create_pipeline(self):
        # The disparity is computed at this resolution, then upscaled to RGB resolution
        monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        # Create pipeline
        pipeline = dai.Pipeline()
        queueNames = []

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        left = pipeline.create(dai.node.MonoCamera)
        right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        rgbOut = pipeline.create(dai.node.XLinkOut)
        disparityOut = pipeline.create(dai.node.XLinkOut)

        rgbOut.setStreamName("rgb")
        queueNames.append("rgb")
        disparityOut.setStreamName("disp")
        queueNames.append("disp")

        #Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setFps(self.internal_fps)
        if self.downscaleColor: camRgb.setIspScale(2, 3)
        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        try:
            calibData = self.device.readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
        except:
            raise
        left.setResolution(monoResolution)
        left.setCamera("left")
        left.setFps(self.internal_fps)
        right.setResolution(monoResolution)
        right.setCamera("right")
        right.setFps(self.internal_fps)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # LR-check is required for depth alignment
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # Linking
        camRgb.isp.link(rgbOut.input)
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        stereo.disparity.link(disparityOut.input)

        print("Pipeline created.")

        return pipeline 

    def next_frame(self):
        self.fps.update()
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["disp"] = None

        queueEvents = self.device.getQueueEvents(("rgb", "disp"))
        # queueEvents = ['rgb','disp']
        # with self.device:
        for queueName in queueEvents:
            packets = self.device.getOutputQueue(queueName).tryGetAll()
            # pdb.set_trace()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        frameRgb = None
        frameDisp = None
        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()

        if latestPacket["disp"] is not None:
            frameDisp = latestPacket["disp"].getFrame()
    
        current_fps = self.fps.get()

        return frameRgb,frameDisp,current_fps
    
    def exit(self):
        self.device.close()


def read_rgbd(internal_fps=35,show=True):
    # Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
    # Otherwise (False), the aligned depth is automatically upscaled to 1080p
    downscaleColor = True
    fps = internal_fps
    # The disparity is computed at this resolution, then upscaled to RGB resolution
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

    # Create pipeline
    pipeline = dai.Pipeline()
    device = dai.Device()
    queueNames = []

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    disparityOut = pipeline.create(dai.node.XLinkOut)

    rgbOut.setStreamName("rgb")
    queueNames.append("rgb")
    disparityOut.setStreamName("disp")
    queueNames.append("disp")

    #Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(fps)
    if downscaleColor: camRgb.setIspScale(2, 3)
    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise
    left.setResolution(monoResolution)
    left.setCamera("left")
    left.setFps(fps)
    right.setResolution(monoResolution)
    right.setCamera("right")
    right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # Linking
    camRgb.isp.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.disparity.link(disparityOut.input)

    # Connect to device and start pipeline
    with device:
        device.startPipeline(pipeline)

        frameRgb = None
        frameDisp = None

        # Configure windows; trackbar adjusts blending ratio of rgb/depth
        rgbWindowName = "rgb"
        depthWindowName = "depth"
        if show:
            cv2.namedWindow(rgbWindowName)
            cv2.namedWindow(depthWindowName)

        fps = FPS()
        while True:
            fps.update()
            latestPacket = {}
            latestPacket["rgb"] = None
            latestPacket["disp"] = None

            queueEvents = device.getQueueEvents(("rgb", "disp"))
            for queueName in queueEvents:
                packets = device.getOutputQueue(queueName).tryGetAll()
                if len(packets) > 0:
                    latestPacket[queueName] = packets[-1]

            if latestPacket["rgb"] is not None:
                frameRgb = latestPacket["rgb"].getCvFrame()
                if show:
                    fps.draw(frameRgb, orig=(50,50), size=1, color=(240,180,100))
                    cv2.imshow(rgbWindowName, frameRgb)
            else:
                print("missing RGB frame!")

            if latestPacket["disp"] is not None:
                frameDisp = latestPacket["disp"].getFrame()
                if show:
                    maxDisparity = stereo.initialConfig.getMaxDisparity()
                    # Optional, extend range 0..95 -> 0..255, for a better visualisation
                    if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
                    # Optional, apply false colorization
                    if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
                    frameDisp = np.ascontiguousarray(frameDisp)
                    fps.draw(frameDisp, orig=(50,50), size=1, color=(240,180,100))
                    cv2.imshow(depthWindowName, frameDisp)
            else:
                print("missing DISP frame!")
            
            if show:
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                current_fps = fps.get()
        

if __name__ == '__main__':
    read_rgbd(internal_fps=35,show=True)