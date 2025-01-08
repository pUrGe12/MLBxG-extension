# Using the pre-trained model on MLB data

Found the cool github repo linked previously. Here I'll test that out and see if the models are any good. 

I have directly used the weights given by them.

## It works!

This is running a single image detection using that model

![image](./outputs/baseball_img_3.png)


## Tracking idea

We can split the image into different frames and then apply this model to each frame, get the coordinates of the ball. 

Now we need to be smart here

Sometimes the balls coordinates will be completely wrong (because the detection model will detect something else as the ball too, it happens)
So, we also need to check if the coordinates of the ball in the current frame are radically different from those in the previous frame, then it means that the detection model has identified something else as the ball and hence, we must neglect that frame's data.
Then once we neglect that frames data (this means, we don't add those coordinates in our list), we go to the next frame and again check the coords with the last ones present in our list (that is the one before the neglected frame)

Also, we will only calculate the speed when we find the ball in at least 7 consecutive frames

This is because the video feed I'll be using is around 4 seconds long. The ball moves with the pitcher's hand too and is sometimes picked up. This might go into our coordinates list!

So, in that list, we also need tomaintain a frame count, and a live time count. That is, assuming 30fps, if I have 4 seconds of data it becomes 120 frames in total. Now assuming in we get the baseball detections in frames 2, 4, 10, 20, then 30 to 60 continously and then 65 and fianlly 118, this means the ball was continously travelling for the frames 30 to 60. 

This means 30 frames which imples 1 second (cause 30 frames per second). Hence the ball travelled for 30 seconds and the pitch length is say 60 feet += 10 feet (we'll give a tolerance because the baseball might stop being detected when close to the batter due to same color) then we get an estimate for speed as being in between 70 ft/s to 50 ft/s.