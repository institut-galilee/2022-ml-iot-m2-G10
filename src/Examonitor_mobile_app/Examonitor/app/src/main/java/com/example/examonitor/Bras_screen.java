package com.example.examonitor;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class Bras_screen extends AppCompatActivity {
    private Button button;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_bras_screen);
        button = (Button) findViewById(R.id.button4);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Pour ouvrir la 2eme page
                ouvrirExamBras_Screen();
            }
        });
    }
    public void ouvrirExamBras_Screen() {
        Intent intent = new Intent(this, Exam_bras_screen.class);
        startActivity(intent);
    }
}