# Test files

Workflow 1 - `Non-deeplearning approach`

✔️ Use a background substrator using Gaussian mixture models (which is what cv2 does) and plot contours around the ball.
✔️ Use SORT to track and calculate metrics for the ball

1. Can we use cv2.createBackgroundSubtractorMOG2() to isolate the background and highlight the ball?

- [x] The problem here is that, since the pitcher and the batsman are also moving pretty fast, it gets very hard to isolate the ball.

Workflow 2 - `deeplearning approach`

✔️ Use YOLOv5 fine-tuned on baseball images to isolate the baseball in the buffer video (we can apply some CV techniques to reduce the effects of excessive and changing lightning)
✔️ Use SORT then on the bounding box and calculate metrics

- [x] Fine-tune YOLOv5 on baseball images.

**Get better data!**