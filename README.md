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





```




**YOLOv8** <br>
Compared to yolov5, YOLOv8 is the state-of-the-art detector which can be installed directly from pip.  

#### Installation and environments
```
pip install ultralytics
# yolov8 available after ultralytics installed

# As the output path e.g. in the folder 'runs' could be 


```

#### Training and prediction

```
1.
yolo predict model=<pt_file> source=<data_to_predict> imgsz=<img_size> save_txt=<True_or_False>
e.g.
yolo predict model=/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/yolov5/yolov8_runs/obb/train_smalltest/weights/best.pt source=/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/yolov5/videos/new_det-17.mp4 imgsz=640 save_txt=True
#-- example of obb prediction using new_det-17.mp4 

2.
yolo task=<> mode=<> model=<pretrained_pt_file> data=<custom_dataset_yaml_path> epochs=<> imgsz=<> batch=<> save_txt=<True_or_False>
e.g.1.
yolo task=obb mode=train model=yolov8l-obb.pt data=/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/yolov5/yolov8_datasets/duckiebotsmalltest/datasmall-obb.yaml epochs=100 imgsz=640 batch=8 save_txt=True
#-- examples of obb training 

e.g.2.
yolo task=obb mode=train model=yolov8m-obb.pt data=/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/yolov5/yolov8_datasets/splitcvatbot/data-obb.yaml epochs=80 imgsz=1024 batch=10 save_txt=True
#-- with imgsz=1024 and splitcvatbot datasets for obb training


```

Alternatively by using python script the training processes can also be initiated as shown in: 
1.
https://medium.com/@KaziMushfiq1234/train-yolov8-on-a-custom-dataset-for-object-detection-on-local-machine-using-python-6a402a2968de
2.
https://learnopencv.com/train-yolov8-on-custom-dataset/






## Code

### in folder 'code_calibration'

**cali_videos_mp4.py**<br>
A python script which is used for camera calibration of a given mp4 video with already measured parameters that compose the camera matrix and distortion coefficients as default inputs.

```
## repository to change
1.
cap = cv2.VideoCapture(<video_path>)
e.g.
cap = cv2.VideoCapture('/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/yolov5/runs/detect/exp29/my_video-10.mp4')

2.
out = cv2.VideoWriter(<output_video_path>, fourcc, fps, (width, height))
e.g.
out = cv2.VideoWriter('/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/yolov5/runs/detect/exp29/undistorted_my_video-10.mp4', fourcc, fps, (width, height))



```

**cali_videos.py**<br>
Similar to cali_videos_mp4.py, a python script which is used for calibration of certain frames


```
1.
image_folder = <input_frame_dir/>
e.g.
image_folder = '/media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/code_lane_detection/frames_undistorted/'

2.
output_folder = <output_frame_dir/>
e.g.
output_folder = '/media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/images/calibration/after/for_det_1_new/'

3.
w1,h1 = <frame_width>,<frame_height>
e.g.
w1,h1 = 1280,720

```

**

### 

**speed_rmse.py**
For speed root mean square error data and diagram generation
```
1.
ground_truth_path = <ground_truth_speed_path>
e.g.
ground_truth_path = '/home/ziwei/Documents/speed_output/speed_anno_obb/id_1.txt'

2.
prediction_paths = [<speed_txt_path_object_1>,<speed_txt_path_object_2>,...,<speed_txt_path_object_n>]
e.g.
prediction_paths = ['/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/new_det-17_speed_littleducks_2_new_length.txt', '/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/speed_new_det-17_2_new_length.txt', '/home/ziwei/Documents/speed_output/speed_fromtrain1024ok/id_1.txt','/home/ziwei/Documents/speed_output/speed_smallyellow/id_1.txt']

3.
labels = [<object_1>, <object_2>,...,<object_n>]
e.g.
labels = ['Duckie', 'Bot', 'Bot_obb','Smallyellow']

4.
colors = [<color_object_1>, <color_object_2>,...]
e.g.
colors = ['skyblue', 'orange', 'green','red']
```

**vt_diagram.py**
For speed v-t diagram generation
```
1.
file_paths = [<speed_txt_path_object_1>,<speed_txt_path_object_2>,...,<speed_txt_path_object_n>]
e.g.
file_paths = ['/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/new_det-17_speed_littleducks_2_new_length.txt', '/home/ziwei/Documents/speed_output/speed_anno_obb/id_1.txt','/media/ziwei/PortableSSD/Junpeng/from_git/test_traffic/speed_new_det-17_2_new_length.txt', '/home/ziwei/Documents/speed_output/speed_fromtrain1024ok/id_1.txt'] 

2.
labels = [<object_1>, <object_2>,...,<object_n>]
e.g.
labels = ['Duckie', 'Bot', 'Bot_obb','Smallyellow']
```

**json2txt.py**
As the output of ground-truth annotation in labelme is in json format, it has to be converted to yolo txt format as the first step before sending it into yolo for training process

```
1.
txt_name = <dir_to_output_txt_files/> + json_name[0:-5] + '.txt'
e.g.
txt_name = '/media/ziwei/PortableSSD/Junpeng/from_git/yolov5/duckiebot/datasets/labels/train/' + json_name[0:-5] + '.txt'

2.
name2id= {<object_name>:<object_index>,...}
e.g.
name2id= {'duckiebot':0}

3.
json_floder_path= <dir_to_input_json_files>
e.g.
json_floder_path= '/media/ziwei/PortableSSD/Junpeng/from_git/yolov5/json/my_video_10'
```

## ffmpeg for frames/video convertion

```
1.
ffmpeg -r <frame_rate> -i <png_sequenz> -q:v 1 -c:v libx264 -pix_fmt yuv420p <path_to_mp4>
  #-- to convert some png files into a mp4 video without resolution loss
  O1
   ffmpeg -r 30 -i %04d.PNG -c:v libx264 -pix_fmt yuv420p /media/ziwei/pictures_to_show/videos/9_pick_up/col/col_pick_up_1/anno_col_pick_up_1.mp4




2.
ffmpeg -i <video_dir> -q:v 1 -r <frame_rate> <dir_to_frames>
e.g.1
ffmpeg -i new_det-19.mp4 -q:v 1 -r 30 /media/ziwei/PortableSSD/littleducks_images/new_19_frame%4d.png
#-- for convertion from video to frames without resolution loss


3.
ffmpeg -i input.mp4 -vf "scale=1280:720" output.mp4
#-- to convert the video to a resolution that we want to have such as 1280*720

4.
ffmpeg -i <input_image_file> -vf "scale=1280:720" <output_image_file>
#-- to convert the image file into a resolution that we want to have such as 1280*720

5.
ffmpeg -i <input_mp4_file> -filter:v "setpts=<multiplier>*PTS" <output_mp4_file>
e.g.1.
ffmpeg -i input.mkv -filter:v "setpts=0.5*PTS" output.mkv
#-- to adjust the speed of a video


6.
ffmpeg -i <original_format> -c copy <target_format>
#-- to convert a video into another format
e.g.1.
ffmpeg -i example.mkv -c copy example.mp4



```




