package com.mobed.mlkit.vision.datacollector.facedetector;
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

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.DisplayMetrics;

import com.mobed.mlkit.vision.datacollector.GraphicOverlay;
import com.mobed.mlkit.vision.datacollector.GraphicOverlay.Graphic;
import com.google.mlkit.vision.face.Face;

/**
 * Graphic instance for rendering eye position and Gaze point
 * graphic overlay view.
 */
public class FaceGraphic extends Graphic {
    private String TAG = "MOBED_FaceGraphic";

    private final Paint  rectColor;

    FaceGraphic(GraphicOverlay overlay) {
        super(overlay);

        rectColor = new Paint();
        rectColor.setARGB(128, 255, 0, 0);
    }

    /**
     * Draws the eye positions and gaze point.
     */
    @Override
    public void draw(Canvas canvas) {
        //Log.d(TAG, "Canvas Width: "+canvas.getWidth()+" Height: "+ canvas.getHeight());

        // Draws a circle at the position of the estimated gaze point
        DisplayMetrics dm = getApplicationContext().getResources().getDisplayMetrics();
        //Log.d(TAG, "Display Metric w/h: " + width+"/"+height);
        int width = canvas.getWidth();
        int height = canvas.getHeight();
        canvas.drawRect(0, 0, width, height, rectColor);
    }
}