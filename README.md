# Duckietown object detection and speed estimation
This project is for detection of multiple objects like duckiebot, duckie, yellow point as a mark on the surface of the duckiebot etc. by using YOLOv5 and YOLOv8.<br>
<br></br>
![](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExODJpNzB1N3N3em51eXU4MXh6ODF6cjM2eHYwaHdzcWh3OWE3ZmZnZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/EY3bPBV2zWYak94rJW/giphy.gif)
![](https://media.giphy.com/media/CmyZ39su0u4Shv6WO3/giphy.gif)












## How to use?
Object annotation is one of the prerequisites for learning-based detectors such as YOLOv5 or YOLOv8 to locate and extract the object semantic information from the scene. For that, we use annotation tools e.g. CVAT (for oriented bounding boxes(OBB)) and labelme (for horizontal bounding boxes(HBB))for ground-truth annotation. Afterwards, it can be used for training process in yolov5 or yolov8 so that a custom dataset for one certain kind of object would be generated (e.g. one single dataset for duckiebot detection). Considering the facts that each annotation tool only supports a certain range of data format (not neccessarily yolo or yolo-obb format). Some transformation of data format using OpenCV should therefore be conducted after exporting ground-truth information. 


## YOLO

### YOLOv5
As YOLOv5 is open-source with a repository on github, we could simply clone that repository to local PC and modify the detection code in Python for speed estimation. 

**Training**<br>

```

python3 train.py --img <image_size> --epochs <epoch_num> --data <yaml_file> --weights <pretrained_pt_file_from_yolov5>
# to train a custom dataset by using the yaml file which is modified based on the default version for custom dataset and pretrained yolov5 pt file in the repository
e.g.
python3 train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt

python3 train.py
# or after modifying certain parts in parser settings we can proceed without flags


```

**Detection**<br>

```
python3 detect.py --weights <pt_file_of_custom_dataset> --data duckiebot/duckiebot_parameter.yaml --source videos/new_det-17.mp4 --view-img --hide-conf --save-txt

e.g.
python detect.py --source 0  # webcam
                          img.jpg  # image 
                          vid.mp4  # video
                          screen  # screenshot
                          path/  # directory
                         'path/*.jpg'  # glob
                         'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                         'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
#-- flags can be modified correspondingly for different modes
e.g.
python3 detect.py --weights runs/train/exp7/weights/best.pt --data duckiebot/duckiebot_parameter.yaml --source videos/new_det-17.mp4 --view-img --hide-conf --save-txt
# to detect the duckiebot using the custom dataset on a specific video with confidence rate on the top of bounding boxes hidden and also the txt files with yolo format would be stored along with the video, in which the detection results are visualized 

e.g.
python3 test_double_bots.py --weights yolov5/runs/train/exp7/weights/best.pt --data yolov5/duckiebot/duckiebot_parameter.yaml --source yolov5/videos/new_det-17.mp4 --view-img --hide-conf --save-txt
# for speed estimation with almost the same inputs above

e.g.
python3 detect.py --weights runs/train/exp7/weights/best.pt --data duckiebot/duckiebot_parameter.yaml --source /media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/yolov5/videos/new_det-17_frames --view-img --hide-conf --save-txt
# instead of videos as input we could also use frames

e.g.



```




**YOLOv8** <br>
Compared to yolov5, YOLOv8 is the state-of-the-art detector which can be installed directly from pip.  

```
pip install ultralytics
# yolov8 available after ultralytics installed



```


### Code

**cali_videos_mp4.py**<br>
A python script which is used for camera calibration of a given mp4 video with already measured parameters that compose the camera matrix and distortion coefficients as default inputs.

```
## repository to change

cap = cv2.VideoCapture(<video_path>)
e.g.
cap = cv2.VideoCapture('/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/yolov5/runs/detect/exp29/my_video-10.mp4')



```




example dataset (10GB)!!<br>
[mydataset(kitti00)](https://drive.google.com/file/d/1T1KrqSesag_-sO5D6IOZttP0yGVrRPhi/view?usp=sharing)<br>
More datasets to be released:<br>