/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mobed.mlkit.vision.datacollector.facedetector;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.PointF;

import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.mobed.mlkit.vision.datacollector.GraphicOverlay;
import com.mobed.mlkit.vision.datacollector.InferenceInfoGraphic;
import com.mobed.mlkit.vision.datacollector.LivePreviewActivity;
import com.mobed.mlkit.vision.datacollector.R;
import com.mobed.mlkit.vision.datacollector.VisionProcessorBase;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;


/**
 * Face Detector Demo.
 */
public class FaceDetectorProcessor extends VisionProcessorBase<List<Face>> {
    private static final String TAG = "MOBED_FaceDetector";
    private static int START_MILL = 650;
    private static int END_MILL = 450;
    private static float EYE_OPEN_PROB = 0.0f;
    private int resolution = 224;
    public static Bitmap image;
    private final FaceDetector detector;
    public float leftEyeleft, leftEyetop, leftEyeright, leftEyebottom;
    public float rightEyeleft, rightEyetop, rightEyeright, rightEyebottom;
    private static final float EYE_BOX_RATIO = 1.4f;
    private String basedir;
    private Button myBtn;
    //Button
    private final int button_size = 224;
    private final int left_margin = 53;
    private final int top_margin = 136;
    private RelativeLayout.LayoutParams params;
    private int leftmargin;
    private int topmargin;
    private int save_gazeX;
    private int save_gazeY;
    private static boolean takePicture = false;

    private TextView textView,textView2;
    Context Fcontext;

    public FaceDetectorProcessor(Context context, FaceDetectorOptions options ) {
        super(context);
        this.Fcontext = context;
        Log.v(MANUAL_TESTING_LOG, "Face detector options: " + options);
        detector = FaceDetection.getClient(options);
        basedir = Environment.getExternalStorageDirectory().getPath()+"/CaptureApp/";
        textView = ((Activity)context).findViewById(R.id.textview);
        textView2 = ((Activity)context).findViewById(R.id.textview2);
        myBtn = (Button)((Activity)context).findViewById(R.id.MyButton);
        params = (RelativeLayout.LayoutParams) myBtn.getLayoutParams();
        leftmargin = params.leftMargin + button_size / 2;
        topmargin = params.topMargin + button_size / 2;
        myBtn.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View view) {
                myBtn = (Button) view.findViewById(R.id.MyButton);
                params = (RelativeLayout.LayoutParams) myBtn.getLayoutParams();
                leftmargin = params.leftMargin + button_size / 2;
                topmargin = params.topMargin + button_size / 2;
                Log.d(TAG, leftmargin + "," + topmargin);
                // TODO : takePicture;
                takePicture = true;
                Handler handler = new Handler();
                handler.postDelayed(new Runnable() {
                    long seed = System.currentTimeMillis();
                    Random rand = new Random(seed);

                    public void run() {
                        int num = rand.nextInt(35);
                        int row = num % 7 + 1; //1~7
                        int col = num % 5 + 1; //1~5
                        int topmargin = 100 + (row - 1) * top_margin + button_size * (row - 1);
                        int leftmargin = col * left_margin + button_size * (col - 1);
                        params.topMargin = topmargin;
                        params.leftMargin = leftmargin;
                        myBtn.setLayoutParams(params);
                    }
                }, 1000);
            }
        });
    }

    @Override
    public void stop() {
        super.stop();
        detector.close();
    }

    @Override
    protected Task<List<Face>> detectInImage(InputImage image) {
        return detector.process(image);
    }

    @Override
    protected void onSuccess(@NonNull List<Face> faces, @NonNull GraphicOverlay graphicOverlay) {
        /**
         * TODO
         * MOBED
         * Notice!
         * Real "Left eye" would be "Right eye" in the face detection. Because camera is left and right reversed.
         * And all terms in face detection would follow direction of camera preview image
         * */
        for (Face face : faces) {
            //MOBED
            //This is how you get coordinates, and crop left and right eye
            //Look at https://firebase.google.com/docs/ml-kit/detect-faces#example_2_face_contour_detection for details.
            //We specifically used Eye Contour's point 0 and 8.
            if (face.getRightEyeOpenProbability() != null && face.getLeftEyeOpenProbability() != null) {
                float rightEyeOpenProb = face.getRightEyeOpenProbability();
                float leftEyeOpenProb = face.getLeftEyeOpenProbability();
                Log.d(TAG, "Right Eye open: "+ rightEyeOpenProb+", Left Eye open: "+leftEyeOpenProb);
                if(rightEyeOpenProb<EYE_OPEN_PROB || leftEyeOpenProb <EYE_OPEN_PROB) continue;
            }
            else {
                Log.d(TAG, "Eye open prob is null");
            }
            try {
                List<PointF> leftEyeContour = face.getContour(FaceContour.LEFT_EYE).getPoints();
                List<PointF> rightEyeContour = face.getContour(FaceContour.RIGHT_EYE).getPoints();
                float righteye_leftx = rightEyeContour.get(0).x;
                float righteye_lefty = rightEyeContour.get(0).y;
                float righteye_rightx = rightEyeContour.get(8).x;
                float righteye_righty = rightEyeContour.get(8).y;
                float lefteye_leftx = leftEyeContour.get(0).x;
                float lefteye_lefty = leftEyeContour.get(0).y;
                float lefteye_rightx = leftEyeContour.get(8).x;
                float lefteye_righty = leftEyeContour.get(8).y;
                float righteye_centerx = (righteye_leftx + righteye_rightx)/2.0f;
                float righteye_centery = (righteye_lefty + righteye_righty)/2.0f;
                float lefteye_centerx = (lefteye_leftx + lefteye_rightx)/2.0f;
                float lefteye_centery = (lefteye_lefty + lefteye_righty)/2.0f;
                float lefteyeboxsize = (lefteye_centerx-lefteye_leftx)*EYE_BOX_RATIO;
                float righteyeboxsize = (righteye_centerx-righteye_leftx)*EYE_BOX_RATIO;
                leftEyeleft = lefteye_centerx - lefteyeboxsize;
                leftEyetop = lefteye_centery + lefteyeboxsize;
                leftEyeright = lefteye_centerx + lefteyeboxsize;
                leftEyebottom = lefteye_centery - lefteyeboxsize;
                rightEyeleft = righteye_centerx - righteyeboxsize;
                rightEyetop = righteye_centery + righteyeboxsize;
                rightEyeright = righteye_centerx + righteyeboxsize;
                rightEyebottom = righteye_centery - righteyeboxsize;
                Bitmap leftBitmap=Bitmap.createBitmap(image, (int)leftEyeleft,(int)leftEyebottom,(int)(lefteyeboxsize*2), (int)(lefteyeboxsize*2));
                Bitmap rightBitmap=Bitmap.createBitmap(image, (int)rightEyeleft,(int)rightEyebottom,(int)(righteyeboxsize*2), (int)(righteyeboxsize*2));
                if (leftBitmap.getHeight() > resolution){
                    leftBitmap = Bitmap.createScaledBitmap(leftBitmap, resolution,resolution,false);
                }
                if (rightBitmap.getHeight() > resolution){
                    rightBitmap = Bitmap.createScaledBitmap(rightBitmap, resolution,resolution,false);
                }

                if (leftBitmap.getHeight() < resolution){
                    leftBitmap = Bitmap.createScaledBitmap(leftBitmap, resolution,resolution,false);
                }
                if (rightBitmap.getHeight() < resolution){
                    rightBitmap = Bitmap.createScaledBitmap(rightBitmap, resolution,resolution,false);
                }

                /**
                 * MOBED SaveBitmapToFileCache
                 * Made For Debug Purpose you can save bitmap image to /sdcard/CaptureApp directory
                 * Then check how the bitmap data is.
                 * */
                if (takePicture) {
                    SharedPreferences sf = LivePreviewActivity.getSf();
                    long cur_time = System.currentTimeMillis();
                    long start_time= cur_time-START_MILL;
                    long end_time= cur_time-END_MILL;
                    Log.d(TAG,"Start Time: "+start_time);
                    Log.d(TAG,"End Time: "+end_time);

                    //For the Left Eye
                    String left_path = basedir+"temp/lefteye";
                    String right_path = basedir+"temp/righteye";
                    File left_directory = new File(left_path);
                    File right_directory = new File(right_path);
                    File[] left_files = left_directory.listFiles();
                    File[] right_files = right_directory.listFiles();
                    if(left_files!=null) {
                        for (int i = 0; i < left_files.length; i++) {
                            String right_file_name = right_files[i].getName();
                            String left_file_name = left_files[i].getName();
                            //left_file_name = time+","+save_gazeX+","+save_gazeY+","+rotationData+","+gyroData+","+acceleroData+".jpg";
                            String[] array = left_file_name.split(",");
                            //time
                            long file_time = Long.parseLong(array[0]);
                            if (file_time >= start_time && file_time <= end_time) {
                                //gaze point
                                String save_gazeX = array[1];
                                String save_gazeY = array[2];
                                //rotation vector
                                String pitch = array[3];
                                String roll = array[4];
                                //gyroscope
                                String gyroX = array[5];
                                String gyroY = array[6];
                                String gyroZ = array[7];
                                //accelerometer
                                String accelX = array[8];
                                String accelY = array[9];
                                String accelZ = array[10];
                                //rename the bitmap files!
                                int count = sf.getInt("count", 0);
                                String left_save_dir = Environment.getExternalStorageDirectory().getPath() + "/CaptureApp/lefteye";
                                File from = new File(left_path + "/" + left_file_name);
                                File to = new File(left_save_dir + "/" + count + ".jpg");
                                if (!from.renameTo(to)) Log.d(TAG, "Filename rename Failed");
                                Log.d(TAG, "Left Eye Bitmap renamed: " + left_file_name + " to " + count + ".jpg");
                                ;
                                String right_save_dir = Environment.getExternalStorageDirectory().getPath() + "/CaptureApp/righteye";
                                from = new File(right_path + "/" + right_file_name);
                                to = new File(right_save_dir + "/" + count + ".jpg");
                                if (!from.renameTo(to)) Log.d(TAG, "Filename rename Failed");
                                Log.d(TAG, "Right Eye Bitmap renamed: " + right_file_name + " to " + count + ".jpg");
                                appendLog(count + "," + save_gazeX + "," + save_gazeY + "," + pitch + "," + roll + "," + gyroX + "," + gyroY + "," + gyroZ + "," + accelX + "," + accelY + "," + accelZ);
                                LivePreviewActivity.addCount();
                            } else {
                                //delete files
                                Log.d(TAG,"Delete temp files");
                                left_files[i].delete();
                                right_files[i].delete();
                            }
                        }
                    }
                    else {
                        Log.d(TAG,"Files don't exist");
                    }
                    takePicture = false;
                }
                else {
                    //Log Files with milliseconds
                    save_gazeX = leftmargin;
                    save_gazeY = topmargin;
                    long time= System.currentTimeMillis();
                    String gyroData = LivePreviewActivity.getGyroData();
                    String acceleroData = LivePreviewActivity.getAcceleroData();
                    String rotationData = LivePreviewActivity.getOrientation();
                    String file0 = time+","+save_gazeX+","+save_gazeY+","+rotationData+","+gyroData+","+acceleroData+","+".jpg";
                    String file1 = time+".jpg";
                    SaveBitmapToFileCache(leftBitmap,basedir+"temp/lefteye/",file0);
                    SaveBitmapToFileCache(rightBitmap,basedir+"temp/righteye/",file1);
                }
                String count = "Count: "+LivePreviewActivity.getCount();
                textView.setText(count);
                if (InferenceInfoGraphic.getFramesPerSecond() != null) {
                    String fps = "FPS: " + InferenceInfoGraphic.getFramesPerSecond() + " / Latency: " + InferenceInfoGraphic.getLatency();
                    textView2.setText(fps);
                }
            }
            catch (java.lang.IllegalArgumentException e) {
                Log.e(TAG, "This Error mostly occurs when one of the eye's bounding box region is partly out of display");
                e.printStackTrace();
            }
            catch (Exception e){
                e.printStackTrace();
            }
            Log.d(TAG, "Bitmap created");
        }
    }

    /**
     * MOBED
     * Made For Storing Bitmaps
     * */
    public static void SaveBitmapToFileCache(Bitmap bitmap, String strFilePath, String filename) {
        File file = new File(strFilePath);
        if (!file.exists())
            file.mkdirs();
        File fileCacheItem = new File(strFilePath + filename);
        Log.d(TAG, "filename: "+strFilePath + filename);
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(fileCacheItem);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Face detection failed " + e);
    }

    public void appendLog(String text) {
        File logFile = new File(basedir+"log.csv");
        if (!logFile.exists()) {
            try {
                logFile.createNewFile();
                BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
                buf.append("count,gazeX,gazeY,pitch,roll,gyroX,gyroY,gyroZ,accelX,accelY,accelZ");
                buf.newLine();
                buf.close();
            }
            catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try {
            //BufferedWriter for performance, true to set append to file flag
            BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
            buf.append(text);
            buf.newLine();
            buf.close();
        }
        catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
