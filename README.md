# juggle-counter
My brother is getting very good at juggling a soccer ball and he records himself to count how many he has done later. This program will count how many juggles he does for him.
I tried at first tracking the ball only based on shape, but the program would detect multiple erraneous circles and mess up the juggle count.
Then I tried using color HSV and it gave me better results.

![Tracking the ball](Resources/juggling.png)

The y coordinates are the center of the detected ball with the x-axis being the number of frames.
![Graph](Resources/juggle_graph.png)
Each marked X is a peak, when the ball is up and just about to fall down. The number of preaks should be the number of juggles done.

Uses OpenCV (cv2) for the ball detection, SciPy to find the peaks, and Matplotlib to graph.
Run the program by doing `python3 HSV/counter_HSV.py path/to/juggle_video.mp4`
Works with other video files as well, like .MOV.
