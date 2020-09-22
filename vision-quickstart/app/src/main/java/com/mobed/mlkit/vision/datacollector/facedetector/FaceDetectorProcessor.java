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

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.PointF;

import android.graphics.Rect;
import android.os.Environment;
import android.os.Handler;
import android.os.SystemClock;
import android.util.DisplayMetrics;
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

import org.jetbrains.annotations.NotNull;

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

    /**
     * Sizes
     * */
    private final int face_grid_size = 25;
    private final int eye_grid_size = 50;
    private final float EYE_OPEN_PROB = 0.0f;
    private final int resolution = 224;
    private static final float EYE_BOX_RATIO = 1.4f;

    public static Bitmap image;
    private final FaceDetector detector;
    public float leftEyeleft, leftEyetop, leftEyeright, leftEyebottom; // leftEyeleft,leftEyetop,leftEyeright,leftEyebottom,rightEyeleft,rightEyetop,rightEyeright,rightEyebottom
    public float rightEyeleft, rightEyetop, rightEyeright, rightEyebottom;
    private String basedir;
    private Button myBtn;
    private long mLastClickTime;
    /**
     * Device Specific Sizes
     * Button location
     * Row & Column Size
     * */
    private final int ROW_NUM = 20;
    private final int COL_NUM = 10;
    private int button_size,left_margin,top_margin;
    private RelativeLayout.LayoutParams params;
    private int leftmargin;
    private int topmargin;
    private int status_bar_height = 100;
    private int navigation_bar_height = 200;
    private int save_gazeX;
    private int save_gazeY;
    private static boolean takePicture = false;
    private int[][] face_grid, lefteye_grid, righteye_grid;
    private Matrix matrix;
    private DisplayMetrics dm;

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
        Log.d(TAG,"");

        float[] mirrorY = {
                -1, 0, 0,
                0, 1, 0,
                0, 0, 1
        };

        matrix = new Matrix();
        matrix.setValues(mirrorY);
        int resourceId = Fcontext.getResources().getIdentifier("status_bar_height", "dimen", "android");
        if (resourceId > 0) {
            status_bar_height = Fcontext.getResources().getDimensionPixelSize(resourceId);
        }
        resourceId = Fcontext.getResources().getIdentifier("navigation_bar_height", "dimen", "android");
        if (resourceId > 0) {
            navigation_bar_height = Fcontext.getResources().getDimensionPixelSize(resourceId);
        }
        dm = Fcontext.getResources().getDisplayMetrics();
        int grid_x = dm.widthPixels/COL_NUM;
        int grid_y = ((dm.heightPixels - status_bar_height) - navigation_bar_height)/ROW_NUM;
        int grid_padding_x,grid_padding_y,button_size;
        if(grid_x < grid_y) {
            grid_padding_x = grid_x / 10;
            button_size = grid_x - grid_padding_x;
            grid_padding_y = grid_y - button_size;
        }
        else {
            grid_padding_y = grid_y / 10;
            button_size = grid_y - grid_padding_y;
            grid_padding_x = grid_x - button_size;
        }
        this.button_size = button_size;
        this.left_margin = grid_padding_x;
        this.top_margin = grid_padding_y;

        myBtn = (Button)((Activity)context).findViewById(R.id.MyButton);
        params = (RelativeLayout.LayoutParams) myBtn.getLayoutParams();
        params.width = button_size;
        params.height = button_size;
        params.leftMargin = left_margin/2;
        params.topMargin = status_bar_height + top_margin/2;
        myBtn.setLayoutParams(params);
        leftmargin = params.leftMargin + button_size / 2;
        topmargin = params.topMargin + button_size / 2;
        myBtn.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View view) {

                // mis-clicking prevention, using threshold of BUTTON_MOVE_DELAY ms
                long cur_time = System.currentTimeMillis();
                if (cur_time - mLastClickTime < BUTTON_MOVE_DELAY){
                    return;
                }
                mLastClickTime = System.currentTimeMillis();
                myBtn = (Button) view.findViewById(R.id.MyButton);
                params = (RelativeLayout.LayoutParams) myBtn.getLayoutParams();
                leftmargin = params.leftMargin + button_size / 2;
                topmargin = params.topMargin + button_size / 2;
                Log.d(TAG, "Button loc: "+leftmargin + "," + topmargin);
                // TODO : takePicture;
                takePicture = true;
                Handler handler = new Handler();
                handler.postDelayed(new Runnable() {
                    long seed = System.currentTimeMillis();
                    Random rand = new Random(seed);

                    public void run() {
                        takePicture=false;
                        int size = ROW_NUM*COL_NUM;
                        int num = rand.nextInt(size);
                        int row = num / COL_NUM + 1; //1~20
                        int col = num % COL_NUM + 1; //1~10
                        int topmargin = status_bar_height  + top_margin/2 + (row - 1) * top_margin + button_size * (row - 1);
                        int leftmargin = left_margin/2 + (col - 1) * left_margin + button_size * (col - 1);
                        params.topMargin = topmargin;
                        params.leftMargin = leftmargin;
                        myBtn.setLayoutParams(params);
                    }
                }, BUTTON_MOVE_DELAY);
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
        boolean isNoFace = true;
        for (Face face : faces) {
            isNoFace = false;
            if(face != faces.get(0)) {
                //TODO show that face is not fully appearing in the image, cannot detect face or too many faces in image
                graphicOverlay.add(new FaceGraphic(graphicOverlay));
                continue;
            }
            //MOBED
            //This is how you get coordinates, and crop left and right eye
            //Look at https://firebase.google.com/docs/ml-kit/detect-faces#example_2_face_contour_detection for details.
            //We specifically used Eye Contour's point 0 and 8.
            if (face.getRightEyeOpenProbability() != null && face.getLeftEyeOpenProbability() != null) {
                float rightEyeOpenProb = face.getRightEyeOpenProbability();
                float leftEyeOpenProb = face.getLeftEyeOpenProbability();
                Log.d(TAG, "Right Eye open: "+ rightEyeOpenProb+", Left Eye open: "+leftEyeOpenProb);
                if(/* rightEyeOpenProb<EYE_OPEN_PROB || */leftEyeOpenProb <EYE_OPEN_PROB) continue; // in my case my right eye is too small compared to left eye...
            }
            else {
                Log.d(TAG, "Eye open prob is null");
            }
            try {
                /**
                 * Left Eye
                 * */
                List<PointF> leftEyeContour = face.getContour(FaceContour.LEFT_EYE).getPoints();
                float lefteye_leftx = leftEyeContour.get(0).x;
                float lefteye_lefty = leftEyeContour.get(0).y;
                float lefteye_rightx = leftEyeContour.get(8).x;
                float lefteye_righty = leftEyeContour.get(8).y;
                float lefteye_centerx = (lefteye_leftx + lefteye_rightx)/2.0f;
                float lefteye_centery = (lefteye_lefty + lefteye_righty)/2.0f;
                float lefteyeboxsize = (lefteye_centerx-lefteye_leftx)*EYE_BOX_RATIO;
                leftEyeleft = lefteye_centerx - lefteyeboxsize;
                leftEyetop = lefteye_centery + lefteyeboxsize;
                leftEyeright = lefteye_centerx + lefteyeboxsize;
                leftEyebottom = lefteye_centery - lefteyeboxsize;
                Bitmap leftBitmap=Bitmap.createBitmap(image, (int)leftEyeleft,(int)leftEyebottom,(int)(lefteyeboxsize*2), (int)(lefteyeboxsize*2), matrix, false);
                if (leftBitmap.getHeight() > resolution){
                    leftBitmap = Bitmap.createScaledBitmap(leftBitmap, resolution,resolution,false);
                }
                if (leftBitmap.getHeight() < resolution){
                    leftBitmap = Bitmap.createScaledBitmap(leftBitmap, resolution,resolution,false);
                }
                /**
                 * Right Eye
                 * */
                List<PointF> rightEyeContour = face.getContour(FaceContour.RIGHT_EYE).getPoints();
                float righteye_leftx = rightEyeContour.get(0).x;
                float righteye_lefty = rightEyeContour.get(0).y;
                float righteye_rightx = rightEyeContour.get(8).x;
                float righteye_righty = rightEyeContour.get(8).y;
                float righteye_centerx = (righteye_leftx + righteye_rightx)/2.0f;
                float righteye_centery = (righteye_lefty + righteye_righty)/2.0f;
                float righteyeboxsize = (righteye_centerx-righteye_leftx)*EYE_BOX_RATIO;
                rightEyeleft = righteye_centerx - righteyeboxsize;
                rightEyetop = righteye_centery + righteyeboxsize;
                rightEyeright = righteye_centerx + righteyeboxsize;
                rightEyebottom = righteye_centery - righteyeboxsize;
                Bitmap rightBitmap=Bitmap.createBitmap(image, (int)rightEyeleft,(int)rightEyebottom,(int)(righteyeboxsize*2), (int)(righteyeboxsize*2), matrix, false);
                if (rightBitmap.getHeight() > resolution){
                    rightBitmap = Bitmap.createScaledBitmap(rightBitmap, resolution,resolution,false);
                }
                if (rightBitmap.getHeight() < resolution){
                    rightBitmap = Bitmap.createScaledBitmap(rightBitmap, resolution,resolution,false);
                }
                /**
                 * Left, Right Eye Contour for SAGE
                 * */

                if (graphicOverlay.isImageFlipped) {
                    leftEyeleft = graphicOverlay.getWidth() - (leftEyeleft * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                    leftEyeright = graphicOverlay.getWidth() - (leftEyeright * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                    rightEyeleft = graphicOverlay.getWidth() - (rightEyeleft * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                    rightEyeright = graphicOverlay.getWidth() - (rightEyeright * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                } else {
                    leftEyeleft = leftEyeleft * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                    leftEyeright = leftEyeright * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                    rightEyeleft = rightEyeleft * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                    rightEyeright = rightEyeright * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                }
                rightEyebottom = rightEyebottom * graphicOverlay.scaleFactor - graphicOverlay.postScaleHeightOffset;
                leftEyebottom = leftEyebottom * graphicOverlay.scaleFactor - graphicOverlay.postScaleHeightOffset;
                /**
                 * Face
                 * */
                Rect facePos = face.getBoundingBox();
                float faceboxWsize = facePos.right - facePos.left;
                float faceboxHsize = facePos.bottom - facePos.top;
                Bitmap faceBitmap=Bitmap.createBitmap(image, (int)facePos.left,(int)facePos.top,(int)faceboxWsize, (int)faceboxHsize, matrix, false);
                faceBitmap = Bitmap.createScaledBitmap(faceBitmap, resolution,resolution,false);
                float faceCenterX = (facePos.right + facePos.left)/2.0f;
                float faceCenterY = (facePos.bottom + facePos.top)/2.0f;
                float face_X, face_Y;
                if (graphicOverlay.isImageFlipped) {
                    face_X = graphicOverlay.getWidth() - (faceCenterX * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                } else {
                    face_X = faceCenterX * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                }
                face_Y = faceCenterY * graphicOverlay.scaleFactor - graphicOverlay.postScaleHeightOffset;
                /**
                 * Face Grid
                 * */
                int image_width = image.getWidth();
                int image_height = image.getHeight();
                //left, bottom, width, height
                float w_start = Math.round(face_grid_size*(facePos.left/(float)image_width));
                float h_start = Math.round(face_grid_size*(facePos.top/(float)image_height));
                float w_num = Math.round(face_grid_size*((faceboxWsize)/(float)image_width));
                float h_num = Math.round(face_grid_size*((faceboxHsize)/(float)image_height));

                face_grid = new int[face_grid_size][face_grid_size];
                for(int h=0; h<face_grid_size; h++){
                    for(int w=0; w<face_grid_size; w++) {
                        if (w>=w_start && w<=w_start+w_num && h>=h_start && h<=h_start+h_num){
                            face_grid[h][(face_grid_size-1)-w] = 1;
                        }
                        else face_grid[h][(face_grid_size-1)-w] = 0;
                    }
                }
                /**
                 * Eye Grids
                 * Use to use these as inputs, but recognized just using eyes results in better results
                 * */
                //left, bottom, width, height
                w_start = Math.round(eye_grid_size*(leftEyeleft/(float)image_width));
                h_start = Math.round(eye_grid_size*(leftEyebottom/(float)image_height));
                w_num = Math.round(eye_grid_size*((2*lefteyeboxsize)/(float)image_width));
                h_num = Math.round(eye_grid_size*((2*lefteyeboxsize)/(float)image_height));

                lefteye_grid = new int[eye_grid_size][eye_grid_size];
                for(int h=0; h<eye_grid_size; h++){
                    for(int w=0; w<eye_grid_size; w++) {
                        if (w>=w_start && w<=w_start+w_num && h>=h_start && h<=h_start+h_num){
                            lefteye_grid[h][(eye_grid_size-1)-w] = 1;
                        }
                        else lefteye_grid[h][(eye_grid_size-1)-w] = 0;
                    }
                }

                w_start = Math.round(eye_grid_size*(rightEyeleft/(float)image_width));
                h_start = Math.round(eye_grid_size*(rightEyebottom/(float)image_height));
                w_num = Math.round(eye_grid_size*((2*righteyeboxsize)/(float)image_width));
                h_num = Math.round(eye_grid_size*((2*righteyeboxsize)/(float)image_height));

                righteye_grid = new int[eye_grid_size][eye_grid_size];
                for(int h=0; h<eye_grid_size; h++){
                    for(int w=0; w<eye_grid_size; w++) {
                        if (w>=w_start && w<=w_start+w_num && h>=h_start && h<=h_start+h_num){
                            righteye_grid[h][(eye_grid_size-1)-w] = 1;
                        }
                        else righteye_grid[h][(eye_grid_size-1)-w] = 0;
                    }
                }

                /**
                 * MOBED Change Data Collecting MODE
                 * */
                if(MODE==0) {
                    //0: Data Collecting from OnClick time-START_MILL to OnClick time-END_MILL
                    if (takePicture) {
                        long cur_time = System.currentTimeMillis();
                        long start_time = cur_time - START_MILL;
                        long end_time = cur_time - END_MILL;
                        Log.d(TAG, "Start Time: " + start_time);
                        Log.d(TAG, "End Time: " + end_time);

                        //For the Left Eye
                        String left_path = basedir + "temp/lefteye";
                        String right_path = basedir + "temp/righteye";
                        String face_path = basedir + "temp/face";
                        String facegrid_path = basedir + "temp/facegrid";
                        String left_eyegrid_path = basedir + "temp/lefteyegrid";
                        String right_eyegrid_path = basedir + "temp/righteyegrid";
                        File left_directory = new File(left_path);
                        File right_directory = new File(right_path);
                        File face_directory = new File(face_path);
                        File facegrid_directory = new File(facegrid_path);
                        File left_eyegrid_directory = new File(left_eyegrid_path);
                        File right_eyegrid_directory = new File(right_eyegrid_path);
                        File[] left_files = left_directory.listFiles();
                        File[] right_files = right_directory.listFiles();
                        File[] face_files = face_directory.listFiles();
                        File[] facegrid_files = facegrid_directory.listFiles();
                        File[] left_eyegrid_files = left_eyegrid_directory.listFiles();
                        File[] right_eyegrid_files = right_eyegrid_directory.listFiles();
                        if (left_files != null) {
                            for (int i = 0; i < left_files.length; i++) {
                                String right_file_name = right_files[i].getName();
                                String left_file_name = left_files[i].getName();
                                String face_file_name = face_files[i].getName();
                                String facegrid_file_name = facegrid_files[i].getName();
                                String left_eyegrid_file_name = left_eyegrid_files[i].getName();
                                String right_eyegrid_file_name = right_eyegrid_files[i].getName();
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
                                    //euler xyz
                                    String eulerX = array[11];
                                    String eulerY = array[12];
                                    String eulerZ = array[13];
                                    //Face xy
                                    String faceX = array[14];
                                    String faceY = array[15];
                                    //
                                    String leftEyeleft= array[16];
                                    String leftEyetop= array[17];
                                    String leftEyeright= array[18];
                                    String leftEyebottom= array[19];
                                    String rightEyeleft= array[20];
                                    String rightEyetop= array[21];
                                    String rightEyeright= array[22];
                                    String rightEyebottom= array[23];
                                    //rename the bitmap files!
                                    int count = LivePreviewActivity.getCount();
                                    //Left Eye
                                    String left_save_dir = Environment.getExternalStorageDirectory().getPath() + "/CaptureApp/lefteye";
                                    File from = new File(left_path + "/" + left_file_name);
                                    @SuppressLint("DefaultLocale") String file0 = String.format("%05d" , count) + ".jpg";
                                    File to = new File(left_save_dir + "/" + file0);
                                    if (!from.renameTo(to)) Log.d(TAG, "Filename rename Failed");
                                    Log.d(TAG, "Left Eye Bitmap renamed: " + left_file_name + " to " + file0);
                                    //Right Eye
                                    String right_save_dir = Environment.getExternalStorageDirectory().getPath() + "/CaptureApp/righteye";
                                    from = new File(right_path + "/" + right_file_name);
                                    to = new File(right_save_dir + "/" + file0);
                                    if (!from.renameTo(to)) Log.d(TAG, "Filename rename Failed");
                                    Log.d(TAG, "Right Eye Bitmap renamed: " + right_file_name + " to " + file0);
                                    //Face
                                    String face_save_dir = Environment.getExternalStorageDirectory().getPath() + "/CaptureApp/face";
                                    from = new File(face_path + "/" + face_file_name);
                                    to = new File(face_save_dir + "/" + file0);
                                    if (!from.renameTo(to)) Log.d(TAG, "Filename rename Failed");
                                    Log.d(TAG, "face Bitmap renamed: " + face_file_name + " to " + file0);
                                    //Face Grid
                                    String facegrid_save_dir = Environment.getExternalStorageDirectory().getPath() + "/CaptureApp/facegrid";
                                    @SuppressLint("DefaultLocale") String file1 = String.format("%05d" , count) + ".csv";
                                    from = new File(facegrid_path + "/" + facegrid_file_name);
                                    to = new File(facegrid_save_dir + "/" + file1);
                                    if (!from.renameTo(to)) Log.d(TAG, "Filename rename Failed");
                                    Log.d(TAG, "facegrid renamed: " + facegrid_save_dir + " to " + file1);
                                    //Left Eye Grid
                                    String left_eyegrid_save_dir = Environment.getExternalStorageDirectory().getPath() + "/CaptureApp/lefteyegrid";
                                    from = new File(left_eyegrid_path + "/" + left_eyegrid_file_name);
                                    to = new File(left_eyegrid_save_dir + "/" + file1);
                                    if (!from.renameTo(to)) Log.d(TAG, "Filename rename Failed");
                                    Log.d(TAG, "lefteyegrid renamed: " + left_eyegrid_save_dir + " to " + file1);
                                    //Right Eye Grid
                                    String right_eyegrid_save_dir = Environment.getExternalStorageDirectory().getPath() + "/CaptureApp/righteyegrid";
                                    from = new File(right_eyegrid_path + "/" + right_eyegrid_file_name);
                                    to = new File(right_eyegrid_save_dir + "/" + file1);
                                    if (!from.renameTo(to)) Log.d(TAG, "Filename rename Failed");
                                    Log.d(TAG, "righteyegrid renamed: " + right_eyegrid_save_dir + " to " + file1);
                                    //Log
                                    appendLog(count + "," + save_gazeX + "," + save_gazeY + "," + pitch + "," + roll + "," + gyroX + "," + gyroY + "," + gyroZ + "," +
                                            accelX + "," + accelY + "," + accelZ + "," +eulerX + "," + eulerY + "," + eulerZ + "," + faceX + "," + faceY+ ","  +
                                            leftEyeleft + "," + leftEyetop + "," + leftEyeright + "," + leftEyebottom + "," + rightEyeleft + "," + rightEyetop + "," + rightEyeright + "," + rightEyebottom);
                                    LivePreviewActivity.addCount();
                                } else {
                                    //delete files
                                    Log.d(TAG, "Delete temp files");
                                    left_files[i].delete();
                                    right_files[i].delete();
                                    face_files[i].delete();
                                    facegrid_files[i].delete();
                                    left_eyegrid_files[i].delete();
                                    right_eyegrid_files[i].delete();
                                }
                            }
                        } else {
                            Log.d(TAG, "Files don't exist");
                        }
                        takePicture = false;
                    } else {
                        //Log Files with milliseconds
                        save_gazeX = leftmargin;
                        save_gazeY = topmargin;
                        long time = System.currentTimeMillis();
                        String gyroData = LivePreviewActivity.getGyroData();
                        String acceleroData = LivePreviewActivity.getAcceleroData();
                        String rotationData = LivePreviewActivity.getOrientation();
                        String file0 = time + "," + save_gazeX + "," + save_gazeY + "," + rotationData + "," + gyroData + "," + acceleroData + "," +
                                face.getHeadEulerAngleX()+","+face.getHeadEulerAngleY()+","+face.getHeadEulerAngleZ()+","+face_X+","+face_Y+ ","  +
                                leftEyeleft + "," + leftEyetop + "," + leftEyeright + "," + leftEyebottom + "," + rightEyeleft + "," + rightEyetop + "," + rightEyeright + "," + rightEyebottom +",.jpg";
                        String file1 = time + ".jpg";
                        String file2 = time + ".csv";
                        SaveBitmapToFileCache(leftBitmap, basedir + "temp/lefteye/", file0);
                        SaveBitmapToFileCache(rightBitmap, basedir + "temp/righteye/", file1);
                        SaveBitmapToFileCache(faceBitmap, basedir + "temp/face/", file1);
                        gridLog(face_grid,face_grid_size,basedir + "temp/facegrid/"+file2);
                        gridLog(lefteye_grid,eye_grid_size,basedir + "temp/lefteyegrid/"+file2);
                        gridLog(righteye_grid,eye_grid_size,basedir + "temp/righteyegrid/"+file2);
                    }
                }
                else if(MODE==1){
                    // 1: Data Collecting on OnClick time
                    if(takePicture) {
                        save_gazeX = leftmargin;
                        save_gazeY = topmargin;
                        long time = System.currentTimeMillis();
                        int count = LivePreviewActivity.getCount();
                        String gyroData = LivePreviewActivity.getGyroData();
                        String acceleroData = LivePreviewActivity.getAcceleroData();
                        String rotationData = LivePreviewActivity.getOrientation();
                        @SuppressLint("DefaultLocale") String file0 = String.format("%05d" , count) + ".jpg";
                        SaveBitmapToFileCache(leftBitmap, basedir + "lefteye/", file0);
                        SaveBitmapToFileCache(rightBitmap, basedir + "righteye/", file0);
                        SaveBitmapToFileCache(faceBitmap, basedir + "face/", file0);
                        @SuppressLint("DefaultLocale") String file1 = String.format("%05d" , count) + ".csv";
                        gridLog(face_grid,face_grid_size,basedir + "facegrid/"+file1);
                        gridLog(lefteye_grid,eye_grid_size,basedir + "lefteyegrid/"+file1);
                        gridLog(righteye_grid,eye_grid_size,basedir + "righteyegrid/"+file1);
                        appendLog(count + "," + save_gazeX + "," + save_gazeY + "," + rotationData + "," + gyroData + "," + acceleroData+","+
                                face.getHeadEulerAngleX()+","+face.getHeadEulerAngleY()+","+face.getHeadEulerAngleZ() + "," + face_X + "," + face_Y + ","  +
                                leftEyeleft + "," + leftEyetop + "," + leftEyeright + "," + leftEyebottom + "," + rightEyeleft + "," + rightEyetop + "," + rightEyeright + "," + rightEyebottom);
                        LivePreviewActivity.addCount();
                        takePicture=false;
                    }
                }else if(MODE==2){
                    // 1: Data Collecting on OnClick time
                    if(takePicture) {
                        long cur_time = System.currentTimeMillis();
                        if(cur_time - mLastClickTime < END_MILL) {
                            save_gazeX = leftmargin;
                            save_gazeY = topmargin;
                            long time = System.currentTimeMillis();
                            int count = LivePreviewActivity.getCount();
                            String gyroData = LivePreviewActivity.getGyroData();
                            String acceleroData = LivePreviewActivity.getAcceleroData();
                            String rotationData = LivePreviewActivity.getOrientation();
                            @SuppressLint("DefaultLocale") String file0 = String.format("%05d", count) + ".jpg";
                            SaveBitmapToFileCache(leftBitmap, basedir + "lefteye/", file0);
                            SaveBitmapToFileCache(rightBitmap, basedir + "righteye/", file0);
                            SaveBitmapToFileCache(faceBitmap, basedir + "face/", file0);
                            @SuppressLint("DefaultLocale") String file1 = String.format("%05d", count) + ".csv";
                            gridLog(face_grid, face_grid_size, basedir + "facegrid/" + file1);
                            gridLog(lefteye_grid, eye_grid_size, basedir + "lefteyegrid/" + file1);
                            gridLog(righteye_grid, eye_grid_size, basedir + "righteyegrid/" + file1);
                            appendLog(count + "," + save_gazeX + "," + save_gazeY + "," + rotationData + "," + gyroData + "," + acceleroData + "," +
                                    face.getHeadEulerAngleX() + "," + face.getHeadEulerAngleY() + "," + face.getHeadEulerAngleZ() + "," + face_X + "," + face_Y + ","  +
                                    leftEyeleft + "," + leftEyetop + "," + leftEyeright + "," + leftEyebottom + "," + rightEyeleft + "," + rightEyetop + "," + rightEyeright + "," + rightEyebottom);
                            LivePreviewActivity.addCount();
                        }
                    }
                }
                String count = "Count: " + LivePreviewActivity.getCount();
                textView.setText(count);
                if (InferenceInfoGraphic.getFramesPerSecond() != null) {
                    String fps = "FPS: " + InferenceInfoGraphic.getFramesPerSecond() + " |  Latency: " + InferenceInfoGraphic.getLatency();
                    textView2.setText(fps);
                }
            }
            catch (java.lang.IllegalArgumentException e) {
                Log.e(TAG, "This Error mostly occurs when one of the eye's bounding box region is partly out of display");
                graphicOverlay.add(new FaceGraphic(graphicOverlay));
                e.printStackTrace();
            }
            catch (Exception e){
                graphicOverlay.add(new FaceGraphic(graphicOverlay));
                e.printStackTrace();
            }
            Log.d(TAG, "Bitmap created");
        }
        if(isNoFace){
            graphicOverlay.add(new FaceGraphic(graphicOverlay));
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
        Log.d(TAG,"Face detection failed");
    }

    public void gridLog(int grid [][], int size, String path){
        File logFile = new File(path);
        if (!logFile.exists()) {
            try {
                logFile.createNewFile();
                BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
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
            for(int h=0; h<size; h++){
                String temp = null;
                for(int w=0; w<size; w++) {
                    if (w==0) temp = Integer.toString(grid[h][w])+",";
                    else if(w==size-1) temp += Integer.toString(grid[h][w]);
                    else temp += Integer.toString(grid[h][w])+",";
                }
                buf.append(temp);
                buf.newLine();
            }
            buf.close();
        }
        catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    public void appendLog(String text) {
        File logFile = new File(basedir+"log.csv");
        if (!logFile.exists()) {
            try {
                logFile.createNewFile();
                BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
                buf.append("count,gazeX,gazeY,pitch,roll,gyroX,gyroY,gyroZ,accelX,accelY,accelZ,eulerX,eulerY,eulerZ,faceX,faceY,leftEyeleft,leftEyetop,leftEyeright,leftEyebottom,rightEyeleft,rightEyetop,rightEyeright,rightEyebottom");
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
