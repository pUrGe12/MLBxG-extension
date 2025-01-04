# Testing YOLOv5

1. Create a Manual dataset (painful)

🚀 Create a Roboflow dataset (not as painful) 

- [x] Test bounding box for a static image in the wild 
- [x] Test boudning box for a video

The problem here is that, I need to keep the threshold confidence very low for detections to happen (like 12%ish). Even then, the model doesn't detect baseballs sometimes.

**The likely issue is that, the images we've used to train are more focused on the baseball, rather than happening in a real game. Find better images and train it again**

- [ ] Create a new dataset and train the model again

## New Idea

- [x] What we can do is, before detection, tweak each frame of the video to match the training and testing dataset. That will definetly increase the confidence rate.

**Not working**

# Training

These are the steps you need to follow to train your YOLOv5s model.


Clone the YOLOv5 repository

		git clone https://github.com/ultralytics/yolov5.git
		cd yolov5

Install the dependencies

		pip install -r requirements.txt

Head over to a [roboflow dataset](https://universe.roboflow.com/yolotest1/yolov5test2). Download the dataset in the `YOLO v5 PyTorch` format. You might get a download link as below. Execute the script where you want to save your dataset. 

		curl -L "https://universe.roboflow.com/ds/lPAYhwr8hN?key=o2ef7xqXid" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

Ensure that it has a `train` and `test` directories with **images** and **labels** inside them.

> [!IMPORTANT]
> Then edit the `data.yaml` file and update the paths to the 

# Testing

Use the python codes present [here](./MoreTestingYoLoAndTracking/flow.py) to test if the model is a good fit or not.