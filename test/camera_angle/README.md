## Idea

Use a single camera angle, that is where the batsman is relative to the bowler. The "height" camera angle we can fix using trial and error based on the values we know for the ball's speed.

This is considering that the baseball pitches are ideally shot from a same angle all the time. That makes life a little easier for us.

## Finding the angle

So we detect the batsman's base and the pitcher's mount. This is because they are relatively stable during the throwing of a pitch. Then find the center of their bounding boxes, that marks their position the screen.

Use screen's dimensions to figure out the angle between the pitcher and the batsman relative to the normal that runs right along the center of the screen, top to the bottom.

## Calculation

So, the pitch distance should change. How exactly? We'll have to figure out the calculations!