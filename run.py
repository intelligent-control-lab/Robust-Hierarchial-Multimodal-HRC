import os,sys
import cv2
import numpy as np
import depthai as dai
from pathlib import Path
import pdb
import marshal
import time
import argparse
import pickle
import torch
import multiprocessing
import rospy
from std_msgs.msg import String
import shutil

FILE_DIR = Path(__file__).parent
# sys.path.append(f"{FILE_DIR}/controller")
# from move_and_grip import receiver
sys.path.append(f'{FILE_DIR}/depthai_blazepose')
from BlazeposeRenderer import BlazeposeRenderer
from BlazeposeDepthaiEdge_module_outside import BlazeposeDepthaiModule
sys.path.append(f'{FILE_DIR}/traj_intention')
from predict import IntentionPredictor
from Dataset import INTENTION_LIST

def get_distance(detection):
    return (detection.spatialCoordinates.x**2+detection.spatialCoordinates.y**2+detection.spatialCoordinates.z**2)**0.5

def camera_to_world(X, R= np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32),\
                     t=0):
    return qrot(np.tile(R, (*X.shape[:-1], 1)), X) + t

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by 四元数quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = np.cross(qvec, v, len(q.shape) - 1)
    uuv = np.cross(qvec, uv, len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv))

def get_intention(index):
    for key,value in INTENTION_LIST.items():
        if value == index:
            return key
    return "no_action"

def send_intention_to_ros(intention):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('intention', anonymous=True)
    rospy.loginfo(intention)
    pub.publish(intention)


def intention_sender(args):
    show = args.show
    task = args.task
    syncNN = args.syncNN
    frame_size = args.seq_len
    send_window = args.send_window
    video = args.video
    ROOT_DIR = f'{FILE_DIR}/human_traj/{task[:-3]}'
    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)

    pipeline = dai.Pipeline()
    # define sources
    camRGB = pipeline.createColorCamera()
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    pose_manip = pipeline.createImageManip()

    # define in/outputs
    xoutDetection = pipeline.create(dai.node.XLinkOut)
    xoutDetection.setStreamName("Detection")
    xoutRGB = pipeline.create(dai.node.XLinkOut)
    xoutRGB.setStreamName("RGB")
    xoutBlazepose = pipeline.create(dai.node.XLinkOut)
    xoutBlazepose.setStreamName("Blazepose")

    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("MaskedFrame")

    # define properties
    internal_fps = 30
    DET_INPUT_SIZE = (300,300)
    det_manip = pipeline.createImageManip()
    det_manip.initialConfig.setResize(DET_INPUT_SIZE[0], DET_INPUT_SIZE[1])
    det_manip.initialConfig.setKeepAspectRatio(False)
    camRGB.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRGB.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRGB.setInterleaved(False)
    camRGB.setFps(internal_fps)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")

    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(False)
    stereo.setLeftRightCheck(True)
    stereo.setConfidenceThreshold(230)
    stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

    spatialDetectionNetwork.setBlobPath(detection_nnPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRGB.preview.link(det_manip.inputImage)
    det_manip.out.link(spatialDetectionNetwork.input)
        
    spatialDetectionNetwork.out.link(xoutDetection.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    if syncNN:
        spatialDetectionNetwork.passthrough.link(xoutRGB.input)
    else:
        camRGB.preview.link(xoutRGB.input)
    
    # set blazepose module
    xyz = True
    internal_frame_height = 450
    no_pos_estimate = False
    blazepose_model = BlazeposeDepthaiModule(input_src="rgb", 
                        pd_model=None,
                        lm_model=None,
                        smoothing=True,   
                        xyz=xyz,            
                        crop=False,
                        internal_fps=internal_fps,
                        internal_frame_height=internal_frame_height,
                        force_detection=False,
                        stats=True,
                        trace=False,
                        no_pos_estimate=no_pos_estimate)
    img_w,img_h = blazepose_model.set_pipeline(pipeline,camRGB,stereo,xinFrame,xoutBlazepose)

    show_3d = False
    renderer = BlazeposeRenderer(
                blazepose_model, 
                show_3d=show_3d, 
                output=None)

    device = dai.Device()
    device.startPipeline(pipeline) 
    calib_data = device.readCalibration()
    calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.CAM_A)
    print(f"RGB calibration lens position: {calib_lens_pos}")
    camRGB.initialControl.setManualFocus(calib_lens_pos)
    
    qDetection = device.getOutputQueue("Detection")
    qRgb = device.getOutputQueue("RGB")
    qBlazepose = device.getOutputQueue("Blazepose")
    inQ = device.getInputQueue("MaskedFrame")
    
    counter = 0
    fps = 0
    startTime = time.monotonic()
    color = (255, 255, 255)
    traj = []
    #  定义编解码器并创建VideoWriter对象
    if video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        pose_fps = 8
        video_out = cv2.VideoWriter(f'{ROOT_DIR}/{task}_camera_out.mp4', fourcc, pose_fps, (img_w,img_h))
    frame_count = 0
    traj_queue = []
    intention_queue = []
    righthand_queue = []
    lefthand_queue = []
    Predictor = IntentionPredictor(model_type=args.model_type)
    old_upperbody = 0
    old_intention = None
    if os.path.exists(f'{ROOT_DIR}/images{task[-3:]}'):
        # If it exists, remove the entire directory and its contents
        shutil.rmtree(f'{ROOT_DIR}/images{task[-3:]}')
    os.makedirs(f'{ROOT_DIR}/images{task[-3:]}')
    while True:
        frame = qRgb.get().getCvFrame()
        
        if qDetection.has():
            detections = qDetection.get()
            detections = detections.detections

            height = frame.shape[0] 
            width  = frame.shape[1]
            nearest_dist = np.inf
            nearest_person = None
            send_flag = 0
            for detection in detections:
                if detection.label == 15 and detection.confidence > 0.6 and \
                    get_distance(detection) < 2500: # TODO
                        send_flag = 1
                        if detection.spatialCoordinates.z < nearest_dist:
                            nearest_dist = detection.spatialCoordinates.z
                            nearest_person = detection
            
            if send_flag == 0:
                # print(f'detection{counter}')
                for detection in detections:
                    if detection.label == 15 and detection.confidence > 0.6:
                        x1 = int(detection.xmin * width)
                        x2 = int(detection.xmax * width)
                        y1 = int(detection.ymin * height)
                        y2 = int(detection.ymax * height)
                        label = 'person'
                        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
                if show:
                    cv2.imshow("preview", frame)

                if cv2.waitKey(1) == ord('q'):
                    break
        
            elif send_flag == 1: # send to pose estimation model
                assert nearest_person
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                # pdb.set_trace()
                x1 = int(max((nearest_person.xmin-0.05),0) * width)
                x2 = int(min((nearest_person.xmax+0.05),1) * width)
                y1 = int(max((nearest_person.ymin-0.05),0) * height)
                y2 = int(min((nearest_person.ymax+0.05),1) * height)
                cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

                img = dai.ImgFrame()
                img.setData(masked_frame.transpose(2,0,1).flatten())
                img.setTimestamp(time.monotonic())
                img.setWidth(img_w)
                img.setHeight(img_h)
                img.setType(dai.RawImgFrame.Type.BGR888p)
                inQ.send(img)

                if qBlazepose.has():
                    res = marshal.loads(qBlazepose.get().getData())
                    body = blazepose_model.inference(res)
                    if body:
                        upperbody = np.concatenate((body.landmarks[11:25,:],body.landmarks[0:1,:]),axis=0)
                        body.landmarks = upperbody
                    masked_frame = renderer.draw(masked_frame, body)
                    # print(f'blazepose{counter}')


                    cv2.putText(masked_frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))

                    # save trajectory
                    if body:
                        try:
                            frame_count += 1
                            cv2.putText(masked_frame, "frame: {:.2f}".format(frame_count), (2, frame.shape[0] - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
                            landmarks = body.landmarks_world
                            righthand = landmarks[16]
                            rightelbow = landmarks[14]
                            rightshoulder = landmarks[12]
                            lefthand = landmarks[15]
                            # upperbody = np.concatenate((landmarks[11:25,:],landmarks[0:1,:]),axis=0)
                            upperbody = body.landmarks
                            if len(traj_queue) < frame_size:
                                traj_queue.append(upperbody)
                            else:
                                traj_queue.pop(0)
                                traj_queue.append(upperbody)
                            assert len(traj_queue) <= frame_size, "the length of accumulated traj is longer than intention prediction frame size!"
                            # intention prediction based on learning 
                            intention = None
                            if len(traj_queue)==frame_size: # send to intention prediction module
                                poses = np.array(traj_queue)
                                poses_norm = 2*(poses-poses.min())/(poses.max()-poses.min())
                                poses_world = camera_to_world(poses_norm)
                                poses_world[:, :, 2] -= np.min(poses_world[:, :, 2])
                                inputs = torch.tensor(poses_world.reshape(1,frame_size,-1)).float()
                                # pdb.set_trace()
                                pred_traj,pred_intention = Predictor.predict(inputs,args.restrict)
                                intention = get_intention(pred_intention)
                                # cv2.putText(masked_frame, f"intention:{intention}", (2, frame.shape[0] - 52), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))


                            # intention prediction based on naive coordinates changes
                            # if righthand[0] < -0.5:
                            #     intention = "get long tubes"
                            # elif righthand[0] > 0.3:
                            #     intention = "get short tubes"
                            # elif len(traj_queue)==frame_size:
                            #     change = 0
                            #     for i in range(frame_size-1):
                            #         change += abs(traj_queue[i+1]-traj_queue[i]).sum()
                            #     if change < 1:
                            #         intention = "waiting"
                            #     else:
                            #         intention = ""
                            # else:
                            #     intention = ""
                            # old_upperbody = upperbody
                            righthand_pose = body.xyz/10 + righthand*100
                            righthand_xpose = body.xyz[0]/10 + righthand[0]*100
                            lefthand_pose = body.xyz/10 + lefthand*100
                            lefthand_xpose = body.xyz[0]/10 + lefthand[0]*100
                            if intention:
                                if len(intention_queue) < send_window:
                                    if len(intention_queue) == 0:
                                        intention_queue.append(intention)
                                        righthand_queue.append(righthand_xpose)
                                        lefthand_queue.append(lefthand_xpose)
                                    else:
                                        if intention == intention_queue[-1]:
                                            intention_queue.append(intention)
                                            righthand_queue.append(righthand_xpose)
                                            lefthand_queue.append(lefthand_xpose)
                                        else:
                                            intention_queue = []
                                            righthand_queue = []
                                            lefthand_queue = []
                                            old_intention = "no_action"
                                else:
                                    if intention != old_intention and intention == intention_queue[-1] and intention != "no_action":
                                        if intention == "get_connectors":
                                            if args.outer_restrict == 'working_area':
                                                # if righthand_xpose < -40: 
                                                if min(righthand_queue) < -33:
                                                    send_intention_to_ros(intention)
                                                    old_intention = intention
                                            else:
                                                send_intention_to_ros(intention)
                                                old_intention = intention
                                        elif intention == "get_screws":
                                            if args.outer_restrict == 'working_area':
                                                # if righthand_xpose > 100 or lefthand_xpose > 100: 
                                                if max(righthand_queue) > 80 or max(lefthand_queue) > 80: 
                                                    send_intention_to_ros(intention)
                                                    old_intention = intention
                                            else:
                                                send_intention_to_ros(intention)
                                                old_intention = intention
                                        elif intention == "get_wheels":
                                            if args.outer_restrict == 'working_area':
                                                if max(righthand_queue) > 100 or max(lefthand_queue) > 100: 
                                                    send_intention_to_ros(intention)
                                                    old_intention = intention
                                            else:
                                                send_intention_to_ros(intention)
                                                old_intention = intention
                                    else:
                                        old_intention = intention
                                    intention_queue = []


                            cv2.putText(masked_frame, f"intention:{intention}", (2, frame.shape[0] - 68), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
                            cv2.putText(masked_frame, "right hand x: {:.2f}, y: {:.2f}, z: {:.2f}".format(righthand_pose[0],righthand_pose[1],righthand_pose[2]), (2, frame.shape[0] - 52), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
                            cv2.putText(masked_frame, "left hand x: {:.2f}, y: {:.2f}, z: {:.2f}".format(lefthand_pose[0],lefthand_pose[1],lefthand_pose[2]), (2, frame.shape[0] - 36), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
                            traj.append(body)
                            if not video:
                                cv2.imwrite(f'{ROOT_DIR}/images{task[-3:]}/{frame_count}.png',masked_frame)
                            else:
                                video_out.write(masked_frame)
                        except:
                            pass
                    if show:
                        cv2.imshow("preview", masked_frame)

                    if cv2.waitKey(1) == ord('q'):
                        break
                
                else:
                    continue

        else:
            continue

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
    file = open(f'{ROOT_DIR}/{task}.pkl', 'wb')
    pickle.dump(traj, file)
    file.close()
    if video:
        video_out.release()

    device.close()


if __name__ == '__main__':
    # path
    parentDir = Path(__file__).parent
    detection_nnPath = str((parentDir / Path('./models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())

    parser = argparse.ArgumentParser()
    parser.add_argument('--syncNN', action="store_true",
                    help="sync the output of detection network with the input of pose estimation network")
    parser.add_argument('--show', action="store_true",
                    help="show real time camera video")
    parser.add_argument('--task', default="test",
                    help="name for traj.pkl and camera video")
    parser.add_argument('--seq_len', default=5,
                        help="input frame window")
    parser.add_argument('--send_window', default=3,
                        help="send if intention is consecutively recognized in send_window")
    parser.add_argument('--video', action="store_true",
                    help="save video, else save images")
    parser.add_argument('--restrict', type=str,default="ood",
                        help='four options:[no,working_area,ood,all]') 
    parser.add_argument('--outer_restrict',type=str,default="working_area",
                        help='outer restriction')
    parser.add_argument('--model_type',type=str,default="final_intention",
                        help='two options:[final_intention,final_traj]')
    args = parser.parse_args()

    intention_sender(args)

                                                                                                                                            
                                                                                                                                        
