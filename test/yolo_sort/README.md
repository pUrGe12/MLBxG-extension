# Testing YOLOv5

1. Create a Manual dataset (painful)
🚀 Create a Roboflow dataset (not as painful) 

- [x] Test bounding box for a static image in the wild 
- [x] Test boudning box for a video

The problem here is that, I need to keep the threshold confidence very low for detections to happen (like 12%ish). Even then, the model doesn't detect baseballs sometimes.

**The likely issue is that, the images we've used to train are more focused on the baseball, rather than happening in a real game. Find better images and train it again**

- [ ] Test the calculations

# Training

These are the steps you need to follow to train your YOLOv5s model.

