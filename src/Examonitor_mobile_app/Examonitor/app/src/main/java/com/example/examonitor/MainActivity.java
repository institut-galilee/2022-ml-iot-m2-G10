package com.example.examonitor;

import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import java.io.PrintWriter;

import androidx.appcompat.app.AppCompatActivity;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.PrintWriter;
import java.net.Socket;

public class MainActivity extends AppCompatActivity {
    private Button button;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        button = (Button) findViewById(R.id.button);

        //new Send_data().execute("Données x,y,z changées");



        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                //Pour ouvrir la 2eme page
                ouvrirIntro_Screen();

            }
        });
    }



    public void ouvrirIntro_Screen() {
        Intent intent = new Intent(this, Intro_Screen.class);
        startActivity(intent);
    }
}