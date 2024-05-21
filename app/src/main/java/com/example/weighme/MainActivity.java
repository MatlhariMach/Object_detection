package com.example.Object_detection;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Surface;
import android.view.TextureView;
import android.widget.ImageView;
import androidx.core.content.ContextCompat;
import com.example.weighme.ml.SsdMobilenetV11Metadata1;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    List<String> labels;
    List<Integer> colors = Collections.unmodifiableList(Arrays.asList(
            Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
            Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED));
    Paint paint = new Paint();
    ImageProcessor imageProcessor;
    Bitmap bitmap;
    ImageView imageView;
    CameraDevice cameraDevice;
    Handler handler;
    CameraManager cameraManager;
    TextureView textureView;
    SsdMobilenetV11Metadata1 model;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        get_permission();

        try {
            labels = FileUtil.loadLabels(this, "labels.txt");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        imageProcessor = new ImageProcessor.Builder().add(new ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build();
        try {
            model = SsdMobilenetV11Metadata1.newInstance(this);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        HandlerThread handlerThread = new HandlerThread("videoThread");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());

        imageView = findViewById(R.id.imageView);

        textureView = findViewById(R.id.textureView);
        textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(@NonNull SurfaceTexture p0, int p1, int p2) {
                open_camera();
            }

            @Override
            public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture p0, int p1, int p2) {
            }

            @Override
            public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture p0) {
                return false;
            }

            @Override
            public void onSurfaceTextureUpdated(@NonNull SurfaceTexture p0) {
                bitmap = textureView.getBitmap();
                TensorImage image = TensorImage.fromBitmap(bitmap);
                image = imageProcessor.process(image);

                SsdMobilenetV11Metadata1.Outputs outputs = model.process(image);

                float[] locations = outputs.getLocationsAsTensorBuffer().getFloatArray();
                float[] classes = outputs.getClassesAsTensorBuffer().getFloatArray();
                float[] scores = outputs.getScoresAsTensorBuffer().getFloatArray();


                Bitmap mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                Canvas canvas = new Canvas(mutable);

                int h = mutable.getHeight();
                int w = mutable.getWidth();
                paint.setTextSize(h/15f);
                paint.setStrokeWidth(h/85f);
                int x = 0;
                for (int i = 0; i < scores.length; i++) {
                    x = i * 4;
                    if(scores[i] > 0.5){
                        paint.setColor(colors.get(i));
                        paint.setStyle(Paint.Style.STROKE);
                        canvas.drawRect(new RectF(locations[x+1]*w, locations[x]*h, locations[x+3]*w, locations[x+2]*h), paint);
                        paint.setStyle(Paint.Style.FILL);
                        canvas.drawText(labels.get((int)classes[i]) + " " + scores[i], locations[x+1]*w, locations[x]*h, paint);
                    }
                }

                imageView.setImageBitmap(mutable);
            }
        });

        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        model.close();
    }

    @SuppressLint("MissingPermission")
    private void open_camera(){
        try {
            cameraManager.openCamera(cameraManager.getCameraIdList()[0], new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice p0) {
                    cameraDevice = p0;

                    SurfaceTexture surfaceTexture = textureView.getSurfaceTexture();
                    Surface surface = new Surface(surfaceTexture);

                    CameraCaptureSession.CaptureCallback captureCallback = new CameraCaptureSession.CaptureCallback() {
                        @Override
                        public void onCaptureStarted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, long timestamp, long frameNumber) {
                            super.onCaptureStarted(session, request, timestamp, frameNumber);
                        }

                        @Override
                        public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
                            super.onCaptureCompleted(session, request, result);
                        }

                        @Override
                        public void onCaptureFailed(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull CaptureFailure failure) {
                            super.onCaptureFailed(session, request, failure);
                        }
                    };

                    try {
                        cameraDevice.createCaptureSession(Collections.singletonList(surface), new CameraCaptureSession.StateCallback() {
                            @Override
                            public void onConfigured(@NonNull CameraCaptureSession p0) {
                                try {
                                    CaptureRequest.Builder captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                                    captureRequestBuilder.addTarget(surface);
                                    p0.setRepeatingRequest(captureRequestBuilder.build(), captureCallback, handler);
                                } catch (CameraAccessException e) {
                                    e.printStackTrace();
                                }
                            }

                            @Override
                            public void onConfigureFailed(@NonNull CameraCaptureSession p0) {
                            }
                        }, handler);

                    } catch (CameraAccessException e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice p0) {
                }

                @Override
                public void onError(@NonNull CameraDevice p0, int p1) {
                }
            }, handler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void get_permission(){
        if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(new String[]{android.Manifest.permission.CAMERA}, 101);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED){
            get_permission();
        }
    }
}
