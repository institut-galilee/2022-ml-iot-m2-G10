package com.example.examonitor;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

public class head_screen extends AppCompatActivity {
    private Button button;
    private void startTask(String info) {

        new ConnectServer().execute(info);
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_head_screen);
        button = (Button) findViewById(R.id.btn_debut_exam);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d("DEBUT_EXAM","debut_exam_tete");
                startTask("STARTED_HEAD");

                Intent intent = getPackageManager().getLaunchIntentForPackage("com.pas.webcam");
                if(intent != null){
                    Log.d("START_IP","application ip_webcam est lancée");
                    startActivity(intent);
                }
            }
        });
    }
}