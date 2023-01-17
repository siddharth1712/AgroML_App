package com.example.agroml;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.content.Context;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends AppCompatActivity {

    int REQUEST_IMAGE_CAPTURE = 1;
    int SELECT_PICTURE = 2;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button captureButton = findViewById(R.id.captureButton);
        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                    // Create the File where the photo should go
                    File photoFile = null;
                    try {
                        photoFile = createImageFile();
                    } catch (IOException ex) {
                        // Error occurred while creating the File
                    }
                    // Continue only if the File was successfully created
                    if (photoFile != null) {
                        Uri photoURI = FileProvider.getUriForFile(this,"com.example.android.fileprovider", photoFile);
                        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
                    }
                }
            }
        });

        Button galleryButton = findViewById(R.id.galleryButton);
        galleryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                imageChooser();
            }
        });

    }
    void imageChooser(){
        Intent i = new Intent();
        i.setType("image/*");
        i.setAction(Intent.ACTION_GET_CONTENT);

        startActivityForResult(Intent.createChooser(i,"Select Picture"),SELECT_PICTURE);
        

    }
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);
        Bitmap bitmap = null;
        Module module = null;
        Bitmap bitmap_resized = null;
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            try {
                bitmap = (Bitmap) data.getExtras().get("data");
                bitmap_resized = Bitmap.createScaledBitmap(bitmap,224,224,false);
                module = LiteModuleLoader.load(assetFilePath(this, "model.pt"));

            } catch (IOException e) {
                e.printStackTrace();
            }

            String classname = predict(bitmap_resized,module);
            displayViews(bitmap_resized,classname);
        }
        else if (requestCode == SELECT_PICTURE && resultCode == RESULT_OK){
            Uri selectedImageUri = data.getData();
            if ( selectedImageUri != null) {
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
                    bitmap_resized = Bitmap.createScaledBitmap(bitmap,224,224,false);
                    module = LiteModuleLoader.load(assetFilePath(this, "model.pt"));
                } catch (IOException e) {
                    e.printStackTrace();
                }

                String classname = predict(bitmap_resized, module);
                displayViews(bitmap_resized, classname);
            }
        }
    }

    String predict(Bitmap bitmap, Module module){
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};

        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,mean,std);

        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();

        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }
        String className = PlantVillageClasses.PLANTVILLAGE_CLASSES[maxScoreIdx];

        return className;
    }

    public void displayViews(Bitmap bitmap, String className){
        ImageView imageView = findViewById(R.id.imageView);
        imageView.setImageBitmap(bitmap);
        TextView text_view = findViewById(R.id.classResult);
        text_view.setText(className);
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    String currentPhotoPath;

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(imageFileName,".jpg",storageDir);

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }
}