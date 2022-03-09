package com.example.examonitor;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class Intro_Screen extends AppCompatActivity {
    private Button button;
    private Button button_tete;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_intro_screen);
        button = (Button) findViewById(R.id.button2);
        button_tete = (Button) findViewById(R.id.btn_tete);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Pour ouvrir la 2eme page
                ouvrirIntro_Screen();
            }
        });
        button_tete.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Pour ouvrir la 2eme page
                ouvrirTete_Screen();
            }
        });
    }
    public void ouvrirIntro_Screen() {
        Intent intent = new Intent(this, Bras_screen.class);
        startActivity(intent);
    }

    public void ouvrirTete_Screen() {
        Intent intent = new Intent(this, head_screen.class);
        startActivity(intent);
    }
}