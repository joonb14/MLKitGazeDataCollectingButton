# MLKitGazeDataCollectingButton: Gaze Data Collecting Application
This work is related to  GAZEL framework which is published in <a href="http://www.percom.org/">PerCom 2021(GAZEL: Runtime Gaze Tracking for Smartphones)</a>. The MLKitGazeDataCollectingButton is used to collect and build dataset for training and testing.

### Sample Video
[![S9+](https://img.youtube.com/vi/gcQa1eAydj8/0.jpg)](https://www.youtube.com/watch?v=gcQa1eAydj8)<br>
This work is about collecting Ground-truth Gaze Data.<br>
I tested it on Galaxy Tab S6, Galaxy S9+, Galaxy S10+, Pixel 2 XL, Pixel 3 XL.<br>
Depending on the performance of processor, this collecting app will run at 10-20FPS.<br>
You just have to click and stare the button until button moves to another point on the screen. <br>
I doubt that there will be anybody who is willing to collect 99999 data at once, but just to make sure, you need to pull the frames before you reach the count  limit 100000. If you want to collect <br>
To pull the collected frames, use the command below<br>

```shell
adb pull /sdcard/CaptureApp
```
I created "".nomedia" file  just to make sure collected data doesn't appear on galleries or on other apps.<br>
I provided jupyter notebook file(<b>Tab S6 Data.ipynb</b>) to help you parse the data collected by MLKitGazeDataCollectingButton, and also show you how to  train Gaze Estimation Model used for <a href="https://github.com/joonb14/GAZEL.git"><b>GAZEL</b></a><br>

## Details
This app collects various data that are used for gaze estimation based on papers. <br>
Such as Eye cropped images, Face images, Face grid, Eye grids. <br>
And also collects mobile sensor data such as Gyroscope, Accelerometer, Rotation vector(pitch, roll)<br>
Thanks to Firebase MLKit, we could also collected Euler angles, and Face center position.<br>
### Configuration
Configure the values in <b>FaceDetectorProcessor.java</b>
```java
/**
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
```
As shown above app's default collection mode is MODE=2, and will collect frames starting from the time you touched the button until 750 milliseconds after it.<br>
And the touched button will move to another position after 1 second.<br>
So, try not to blink until the the button moves!<br>
Unless, it will be outlier data that will harm your model accuracy.<br>

### Parsing Collected Data

<a href="https://github.com/joonb14/MLKitGazeDataCollectingButton/blob/master/Data%20parsing.ipynb">Data Parsing.ipynb</a> is about parsing the data collected from the MLKitGazeDataCollectingButton. <br>
Suppose your collected data is stored in <b>/special/jbpark/TabS6/Joonbeom/</b> directory<br>
The directory structure will be like below<br>

```
▼ /special/jbpark/TabS6/Joonbeom/
	▶ face
	▶ facegrid
	▶ lefteye
	▶ lefteyegrid
	▶ righteye
	▶ righteyegrid
	▶ log.csv

```
In face, lefteye, righteye directory there will be extracted frame images by MLKitGazeDataCollectingButton. And in facegrid, lefteyegrid, righteyegrid there will be grid data consists of 0 and 1 (This is for training and testing <a href="https://ieeexplore.ieee.org/document/7780608">iTracker</a> and <a href="https://ieeexplore.ieee.org/document/8669057">GazeEstimator</a>).<br>
The log.csv contains mobile embedded sensing data and software computed values.<br>

```
count,gazeX,gazeY,pitch,roll,gyroX,gyroY,gyroZ,accelX,accelY,accelZ,eulerX,eulerY,eulerZ,faceX,faceY,leftEyeleft,leftEyetop,leftEyeright,leftEyebottom,rightEyeleft,rightEyetop,rightEyeright,rightEyebottom
```
for example, if you use pandas to visualize the log.csv it would look like below.<br>
![image](https://user-images.githubusercontent.com/30307587/109653318-3b9e1180-7ba4-11eb-8bbf-3371ce237182.png)<br>
The frames(face, lefteye, righteye) are stored in zero filed format like <br>

```
00000.jpg	00001.jpg	00002.jpg	00003.jpg	00004.jpg	...
```
The numbers align with the count value in log.csv<br>
I provide notebook file that show the process of making the dataset 

### Training Gaze Estimation Model
<img src="https://user-images.githubusercontent.com/30307587/109145286-a6ff7200-77a5-11eb-86ff-41925981af10.png" width=800/>

As mentioned in the GAZEL paper, we used multiple inputs. The <a href="https://github.com/joonb14/MLKitGazeDataCollectingButton/blob/master/Data%20parsing.ipynb">Data Parsing.ipynb</a> shows how to parse these inputs, and <a href="https://github.com/joonb14/MLKitGazeDataCollectingButton/blob/master/GAZEL.ipynb">GAZEL.ipynb</a> shows how to construct toy model.<br>
We won't provide pre-trained model, and the 10 participants' image data due to the right of publicity.<br>
But if you follow the written instruction, collect your gaze data with <a href="https://github.com/joonb14/MLKitGazeDataCollectingButton">MLKitGazeDataCollectingButton</a> it will take about 30 minutes to collect over 5000 samples. Then use the notebook files to parse & create your own model. You must follow the TFLite conversion guideline before you import your tflite model on <a href="https://github.com/joonb14/GAZEL">GAZEL</a>.
<b>[Note] For the evaluation  we used data collected from 10 participants to train general model and calibrated it by each user's implicitly collected frames by Data Collecting Launcher. Not the toy model in the GAZEL.ipynb file</b>

### [Important] TensorFlow Lite Conversion Guideline!
If you don't follow the guideline you would get errors like <a href="https://github.com/tensorflow/tensorflow/issues/19982"><b>this</b></a> when using TFLite interpreter in the Android device<br>
```
java.lang.NullPointerException: Internal error: Cannot allocate memory for the interpreter: tensorflow/contrib/lite/kernels/conv.cc:191 input->dims->size != N (M != N)
```
This error occurs because of the shape of multiple inputs are not same, specifically dimension of them. I will show you the example with code<br>
```python
# Keras

# Left Eye
input1 = Input(shape=(resolution, resolution,channels), name='left_eye')
# Right Eye
input2 = Input(shape=(resolution, resolution,channels), name='right_eye')
# Facepos
input4 = Input(shape=(2), name='facepos')
#Euler
input3 = Input(shape=(3), name='euler')
# Eye size
input5 = Input(shape=(2), name='left_eye_size')
input6 = Input(shape=(2), name='right_eye_size')
```
In the above code you can see that input1,2's dimension is 3 and others are 1.<br>
This is the reason why you get ```input->dims->size != N (M != N)``` error when you try to use interpreter with multiple inputs composed of different dimension. So to solve this error, you should expand all the inputs' dimension to the largest dimension. See the code below.<br>

```python
# Keras

# Left Eye
input1 = Input(shape=(resolution, resolution,channels), name='left_eye')
# Right Eye
input2 = Input(shape=(resolution, resolution,channels), name='right_eye')
# Facepos
input4 = Input(shape=(1, 1, 2), name='facepos')
#Euler
input3 = Input(shape=(1, 1, 3), name='euler')
# Eye size
input5 = Input(shape=(1, 1, 2), name='left_eye_size')
input6 = Input(shape=(1, 1, 2), name='right_eye_size')
```
I expand all input3,4,5,6 to be 3 dimension. Now we need to reshape the loaded training data to match the input shape<br>
```python
gaze_point = np.load(target+"gaze_point.npy").astype(float)
left_eye = np.load(target+"left_eye.npy").reshape(-1,resolution,resolution,channels)
right_eye = np.load(target+"right_eye.npy").reshape(-1,resolution,resolution,channels)
euler = np.load(target+"euler.npy").reshape(-1,1,1,3)
facepos = np.load(target+"facepos.npy").reshape(-1,1,1,2)
left_eye_right_top = np.load(target+"left_eye_right_top.npy")
left_eye_left_bottom = np.load(target+"left_eye_left_bottom.npy")
right_eye_right_top = np.load(target+"right_eye_right_top.npy")
right_eye_left_bottom = np.load(target+"right_eye_left_bottom.npy")

left_eye_right_top[:,1] = left_eye_right_top[:,1] - left_eye_left_bottom[:,1]
left_eye_right_top[:,0] = left_eye_left_bottom[:,0] - left_eye_right_top[:,0]

right_eye_right_top[:,1] = right_eye_right_top[:,1] - right_eye_left_bottom[:,1]
right_eye_right_top[:,0] = right_eye_left_bottom[:,0] - right_eye_right_top[:,0]

left_eye_size = left_eye_right_top.reshape(-1,1,1,2)
right_eye_size = left_eye_left_bottom.reshape(-1,1,1,2)
```

Now, if you matched the dimensions of inputs, you are ready to go!

### Discussion

I used only S9+ for accuracy testing device. So for generalization it would be better to use normalized inputs for left/right eye sizes, euler x,y,z, facepos.<br>
Also, it would definitely better to use large scale dataset for training feature extraction layer. I realized this after submitting the paper unfortunately... So I would recommend using larger dataset for training CNN layers then substitute the architecture with ours, fine tune the architecture with collected data from MLKitGazeDataCollectingButton.
