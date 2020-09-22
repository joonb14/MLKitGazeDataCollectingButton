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

package com.mobed.mlkit.vision.datacollector;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;

import androidx.core.app.ActivityCompat;
import androidx.core.app.ActivityCompat.OnRequestPermissionsResultCallback;
import androidx.core.content.ContextCompat;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.CompoundButton;
import android.widget.Toast;

import com.google.android.gms.common.annotation.KeepName;
import com.mobed.mlkit.vision.datacollector.R;
import com.mobed.mlkit.vision.datacollector.facedetector.FaceDetectorProcessor;
import com.mobed.mlkit.vision.datacollector.preference.PreferenceUtils;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Live preview demo for ML Kit APIs.
 */
@KeepName
public final class LivePreviewActivity extends AppCompatActivity
        implements OnRequestPermissionsResultCallback,
        OnItemSelectedListener,
        CompoundButton.OnCheckedChangeListener {
    private static final String FACE_DETECTION = "Face Detection";
    private static final String TAG = "MOBED_LivePreview";
    private static final int PERMISSION_REQUESTS = 2;

    private CameraSource cameraSource = null;
    private CameraSourcePreview preview;
    private GraphicOverlay graphicOverlay;
    private String selectedModel = FACE_DETECTION;
    private static SharedPreferences sf;
    private static int count;
    private SensorManager mSensorManager;
    private Sensor mGyroSensor = null;
    private Sensor mAccelerometer = null;
    private Sensor mRotationVector = null;

    private static double gyroX;
    private static double gyroY;
    private static double gyroZ;

    private  static double accX;
    private  static double accY;
    private  static double accZ;

    private  static float pitch;
    private  static float roll;

    //private static final float RADIAN_TO_DEGREE= (float) -57.2958;
    private static final int RADIAN_TO_DEGREE= -57;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");

        setContentView(R.layout.activity_vision_live_preview);

        preview = findViewById(R.id.preview);
        if (preview == null) {
            Log.d(TAG, "Preview is null");
        }
        graphicOverlay = findViewById(R.id.graphic_overlay);
        if (graphicOverlay == null) {
            Log.d(TAG, "graphicOverlay is null");
        }

        List<String> options = new ArrayList<>();
        options.add(FACE_DETECTION);

        if (allPermissionsGranted()) {
            createCameraSource(selectedModel);
        } else {
            getRuntimePermissions();
        }

        sf = getPreferences(Context.MODE_PRIVATE);
        //sf.edit().remove("count").commit();
        count = sf.getInt("count",0);

        createDirectories();

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mGyroSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mRotationVector = mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
    }

    public static int getCount(){
        count = sf.getInt("count",0);
        return count;
    }


    public static SharedPreferences getSf() {
        return sf;
    }

    public static int addCount(){
        count+=1;
        SharedPreferences.Editor editor = sf.edit();
        editor.putInt("count",count);
        editor.commit();
        return count;
    }

    @Override
    public synchronized void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
        // An item was selected. You can retrieve the selected item using
        // parent.getItemAtPosition(pos)
        selectedModel = parent.getItemAtPosition(pos).toString();
        Log.d(TAG, "Selected model: " + selectedModel);
        preview.stop();
        if (allPermissionsGranted()) {
            createCameraSource(selectedModel);
            startCameraSource();
        } else {
            getRuntimePermissions();
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
        // Do nothing.
    }

    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        Log.d(TAG, "Set facing");
        if (cameraSource != null) {
            if (isChecked) {
                cameraSource.setFacing(CameraSource.CAMERA_FACING_FRONT);
            } else {
                cameraSource.setFacing(CameraSource.CAMERA_FACING_BACK);
            }
        }
        preview.stop();
        startCameraSource();
    }

    private void createCameraSource(String model) {
        // If there's no existing cameraSource, create one.
        if (cameraSource == null) {
            cameraSource = new CameraSource(this, graphicOverlay);
        }

        try {
            switch (model) {
                case FACE_DETECTION:
                    Log.i(TAG, "Using Face Detector Processor");
                    FaceDetectorOptions faceDetectorOptions =
                            PreferenceUtils.getFaceDetectorOptionsForLivePreview(this);

                    cameraSource.setMachineLearningFrameProcessor(
                            new FaceDetectorProcessor(this, faceDetectorOptions));
                    break;
                default:
                    Log.e(TAG, "Unknown model: " + model);
            }
        } catch (Exception e) {
            Log.e(TAG, "Can not create image processor: " + model, e);
            Toast.makeText(
                    getApplicationContext(),
                    "Can not create image processor: " + e.getMessage(),
                    Toast.LENGTH_LONG)
                    .show();
        }
    }

    /**
     * Starts or restarts the camera source, if it exists. If the camera source doesn't exist yet
     * (e.g., because onResume was called before the camera source was created), this will be called
     * again when the camera source is created.
     */
    private void startCameraSource() {
        if (cameraSource != null) {
            try {
                if (preview == null) {
                    Log.d(TAG, "resume: Preview is null");
                }
                if (graphicOverlay == null) {
                    Log.d(TAG, "resume: graphOverlay is null");
                }
                preview.start(cameraSource, graphicOverlay);
            } catch (IOException e) {
                Log.e(TAG, "Unable to start camera source.", e);
                cameraSource.release();
                cameraSource = null;
            }
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");
        createCameraSource(selectedModel);
        startCameraSource();
        createDirectories();
        mSensorManager.registerListener(gyroListener, mGyroSensor, SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(acceleroListener, mAccelerometer, SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(rotationListener, mRotationVector, SensorManager.SENSOR_DELAY_GAME);
    }

    /**
     * Stops the camera.
     */
    @Override
    protected void onPause() {
        super.onPause();
        preview.stop();
        mSensorManager.unregisterListener(gyroListener);
        mSensorManager.unregisterListener(acceleroListener);
        mSensorManager.unregisterListener(rotationListener);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraSource != null) {
            cameraSource.release();
        }
    }

    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    this.getPackageManager()
                            .getPackageInfo(this.getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                return false;
            }
        }
        return true;
    }

    private void getRuntimePermissions() {
        List<String> allNeededPermissions = new ArrayList<>();
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                allNeededPermissions.add(permission);
            }
        }

        if (!allNeededPermissions.isEmpty()) {
            ActivityCompat.requestPermissions(
                    this, allNeededPermissions.toArray(new String[0]), PERMISSION_REQUESTS);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        Log.i(TAG, "Permission granted!");
        if (allPermissionsGranted()) {
            createCameraSource(selectedModel);
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    private static boolean isPermissionGranted(Context context, String permission) {
        if (ContextCompat.checkSelfPermission(context, permission)
                == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Permission granted: " + permission);
            return true;
        }
        Log.i(TAG, "Permission NOT granted: " + permission);
        return false;
    }

    public boolean dir_exists(String dir_path) {
        boolean ret = false;
        File dir = new File(dir_path);
        if(dir.exists() && dir.isDirectory())
            ret = true;
        return ret;
    }


    public SensorEventListener gyroListener = new SensorEventListener() {
        public void onAccuracyChanged(Sensor mGyroSensor, int acc) {
        }

        public void onSensorChanged(SensorEvent event) {
            gyroX = event.values[0];
            gyroY = event.values[1];
            gyroZ = event.values[2];
            //SLog.d("MOBED","Gyro: "+gyroX+ " "+gyroY+" "+gyroZ+"rad/s");
        }
    };
    public SensorEventListener acceleroListener = new SensorEventListener() {
        public void onAccuracyChanged(Sensor mGyroSensor, int acc) {
        }

        public void onSensorChanged(SensorEvent event) {
            accX = event.values[0];
            accY = event.values[1];
            accZ = event.values[2];
            //Log.d("MOBED","Accelerometer: "+accX+ " "+accY+" "+accZ+"m/s^2");
        }
    };

    //Copied from https://rosia.tistory.com/128
    public SensorEventListener rotationListener = new SensorEventListener() {
        public void onAccuracyChanged(Sensor mRotationVector, int acc) {
        }

        public void onSensorChanged(SensorEvent event) {
            if(event.values.length>4) {
                //Log.d(TAG,"Rotation Vector event.values[0]: "+event.values[0]+" event.values[1]: "+event.values[1]+" event.values[2]: "+event.values[2]+" event.values[3]: "+event.values[3]);
                checkOrientation(event.values);
            }
        }
    };

    private void checkOrientation(float[] rotationVector) {
        float[] rotationMatrix = new float[9];
        SensorManager.getRotationMatrixFromVector(rotationMatrix, rotationVector);

        final int worldAxisForDeviceAxisX = SensorManager.AXIS_X;
        final int worldAxisForDeviceAxisY = SensorManager.AXIS_Z;



        float[] adjustedRotationMatrix = new float[9];
        SensorManager.remapCoordinateSystem(rotationMatrix, worldAxisForDeviceAxisX,
                worldAxisForDeviceAxisY, adjustedRotationMatrix);

        // Transform rotation matrix into azimuth/pitch/roll
        float[] orientation = new float[3];
        SensorManager.getOrientation(adjustedRotationMatrix, orientation);

        // Convert radians to degrees
        pitch = orientation[1] * RADIAN_TO_DEGREE;
        roll = orientation[2] * RADIAN_TO_DEGREE;
        //Log.d(TAG,"Rotation Vector Pitch: "+pitch+" Roll: "+roll);
    }

    public static String getGyroData(){
        return gyroX+ ","+gyroY+","+gyroZ;
    }
    public static String getAcceleroData(){
        return accX+ ","+accY+","+accZ;
    }
    public static String getOrientation(){
        return pitch+ ","+roll;
    }


    private final void createDirectories(){
        //MOBED
        String dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/lefteye";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/righteye";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/face";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/facegrid";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/lefteyegrid";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/righteyegrid";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/temp";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/temp/lefteye";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/temp/righteye";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/temp/face";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/temp/facegrid";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/temp/lefteyegrid";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
        dir_path = Environment.getExternalStorageDirectory() + "/CaptureApp/temp/righteyegrid";
        if (!dir_exists(dir_path)){
            File directory = new File(dir_path);
            if(!directory.mkdirs()){
                Log.e(TAG, "Cannot create Directory "+dir_path);
            }
        }
    }
}
