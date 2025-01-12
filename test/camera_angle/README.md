## Idea

Use a single camera angle, that is where the batsman is relative to the bowler. The "height" camera angle we can fix using trial and error based on the values we know for the ball's speed.

This is considering that the baseball pitches are ideally shot from a same angle all the time. That makes life a little easier for us.

## Finding the angle

So we detect the batsman's base and the pitcher's mount. This is because they are relatively stable during the throwing of a pitch. Then find the center of their bounding boxes, that marks their position the screen.

Use screen's dimensions to figure out the angle between the pitcher and the batsman relative to the normal that runs right along the center of the screen, top to the bottom.

## Calculation

Assuming, 

1. The camera will always be pointed such that the pitcher is player lower than the catcher in the screen.
2. The camera's central point will the normal to the screen (that is, a line passing down the middle, splitting the screen width into two)

let the line made by the catcher and the pitcher make an angle $\alpha$ with the screen's normal. This means $\alpha$ can be calculated using the following

$$
\alpha = tan^{-1} \left(\frac{abs \left(x_2-x_0 \right)}{abs(y')+y_2}\right)
$$

or,

$$
\alpha = tan^{-1} \left( \frac{abs(x_2-x_0)}{y_2 - y'} \right)
$$

where, ($$x_2$$, $$y_2$$) are coordinates of the pitcher and ($$x_1$$, $$y_1$$) are the coordinates of the catcher. This is found using the yolo models we have and we take average of the center of the bounding box for the picher and catcher for each frame. 

$$x_0, y'$$ is the coordinate of the point of intersection of the line joining the catcher and the pitcher to the central normal. $$y'$$ is given by,

$$
y' = \frac{(y_2-y_1)}{x_2-x_1} (x_0 - x_1) + y_1
$$

We'll use one of the two definitions of $\alpha$ depending on whether $$y'$$ happens to be positive or negative. According to the screen's layout, the top left corner is the origin and the X axis is along the right and Y axis perpendicularly downwards. 

If $$y'$$ is positive we use the 2nd definition (because that means it was within the screen itself), and if negative then the first one.

---

After finding the $\alpha$ value we calculate the actual speed. Now think about rotating the line with pitcher and the catcher by this angle $\alpha$ such that it lines up with the screen's normal. This means we shift the camera by the same angle to the right. 

Now, if the real speed of the ball is $$V_r$$ and the speed we calculate is $$V_app$$ then the relation between them becomes

$$
V_app = V_r * sin(\alpha)
$$

and thus,

$$
V_r = V_app * csc(\alpha)
$$

We calculate this and we're done!