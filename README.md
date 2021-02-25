# MLKitGazeDataCollectingButton
### Sample Video
[![S9+](https://img.youtube.com/vi/gcQa1eAydj8/0.jpg)](https://www.youtube.com/watch?v=gcQa1eAydj8)<br>
This work is about collecting Ground-truth Gaze Data.<br>
I tested it on Galaxy Tab S6, Galaxy S9+, Galaxy S10+, Pixel 2 XL, Pixel 3 XL.<br>
Depending on the performance of processor, this collecting app will run at 10-20FPS.<br>
You just have to click and stare the button until button moves to another point on the screen. <br>
I doubt that there will be anybody who is willing to collect 99999 data at once, but just to make sure, you need to pull the frames before you reach the count  limit 100000.<br>
To pull the collected frames, use the command below<br>
<pre><code>adb pull /sdcard/CaptureApp
</code></pre>
I created "".nomedia" file  just to make sure collected data doesn't appear on galleries or on other apps.<br>
I provided jupyter notebook file(<b>Tab S6 Data.ipynb</b>) to help you parse the data collected by MLKitGazeDataCollectingButton, and also show you how to  train Gaze Estimation Model used for <a href="https://github.com/joonb14/GAZEL.git"><b>GAZEL</b></a><br>
## Details
This app collects various data that are used for gaze estimation based on papers. <br>
Such as Eye cropped images, Face images, Face grid, Eye grids. <br>
And also collects mobile sensor data such as Gyroscope, Accelerometer, Rotation vector(pitch, roll)<br>
Thanks to Firebase MLKit, we could also collected Euler angles, and Face center position.<br>
### Configuration
Configure the values in <b>FaceDetectorProcessor.java</b>
<pre><code>/**
* Change Data Collection Mode by MODE Value
*/
private int MODE = 2;
private final int START_MILL = 450;
private final int END_MILL = 750;
private final int BUTTON_MOVE_DELAY = 1000;
/**
* 0: Data Collecting from OnClick time-START_MILL to OnClick time-END_MILL
* 1: Data Collecting on OnClick time
* 2: Data Collecting from OnClick time+START_MILL to OnClick time+END_MILL
* */
</code></pre>
As shown above app's default collection mode is MODE=2, and will collect frames starting from the time you touched the button until 750 milliseconds after it.<br>
And the touched button will move to another position after 1 second.<br>
So, try not to blink until the the button moves!<br>
Unless, it will be outlier data that will harm your model accuracy.<br>