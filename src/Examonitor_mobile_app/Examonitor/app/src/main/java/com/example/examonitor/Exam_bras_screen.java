package com.example.examonitor;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
// Import pour envoyer les données vers python


public class Exam_bras_screen extends AppCompatActivity {
    private Button button_fin;
    TextView textInfoex;
    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private double accelerationActuelle;
    private double accelerationPrecedente;
    //ConnectServer cs = new ConnectServer();
    private void startTask(String position) {

        new ConnectServer().execute(position);
    }

    private void finExamenBras(){
        Log.d("FIN_EXAM_FIRED","yes it has been");
        startTask("L'étudiant a terminé son examen.");
    }


    private SensorEventListener sensorEventListener = new SensorEventListener() {
        String positions_suspectes = new String("Bras");
        int nb_valeurs_suspectes = 0;

        @Override
        public void onSensorChanged(SensorEvent sensorEvent) {
            //Log.d("ONSC","onSensorChanged has been fired mgl");
            float x = sensorEvent.values[0];
            float y = sensorEvent.values[1];
            float z = sensorEvent.values[2];

            accelerationActuelle = Math.sqrt((x*x+y*y+z*z));

            double changement_acceleration = Math.abs(accelerationActuelle - accelerationPrecedente);


           /* txt_accel_curr.setText("x actuelle : "+ Float.toString(x)+"y actuelle"+ Float.toString(y)
                    +"z actuelle"+ Float.toString(z));

            txt_accel_prec.setText("Accéleration précedente : "+ accelerationPrecedente);
            //txt_accel.setText("Accéleration variation  : "+ changement_acceleration);*/
            accelerationPrecedente = accelerationActuelle;

            if(changement_acceleration > 2.0){

                String Newligne=System.getProperty("line.separator");
                x = (float) (Math.abs(x)+ (Math.random() * (2)));
                y =  (float) (Math.abs(y)+ (Math.random() * (5)));
                z =  (float) (Math.abs(z)+ (Math.random() * (7)));
                positions_suspectes = positions_suspectes + Newligne + Float.toString(x)+" "+ Float.toString(y)+" "+ Float.toString(z);
                nb_valeurs_suspectes++;
                Log.d("NBVS", Integer.toString(nb_valeurs_suspectes));
                if(nb_valeurs_suspectes == 10 ){
                    //cs.execute(positions_suspectes);
                    //startTask(positions_suspectes);
                    startTask(positions_suspectes);
                    positions_suspectes = new String("Bras");
                    nb_valeurs_suspectes = 0;
                    Log.d("ENVOYER",positions_suspectes);
                }
            }


        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int i) {

        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_exam_bras_screen);
        textInfoex = findViewById(R.id.textInfoex);


        mSensorManager = (SensorManager)getSystemService(SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        button_fin = (Button) findViewById(R.id.btn_fin_examen);

        startTask("STARTED_ARM");

        //new Send_data().execute("Données x,y,z changées");



        button_fin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                //Pour ouvrir la 2eme page
                finExamenBras();

            }
        });

    }
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(sensorEventListener, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
    }

    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(sensorEventListener);
    }

    private class ConnectServer2 extends AsyncTask<String,Void,String> {
        Socket soc;
        PrintWriter pw;
        String response;

        @Override
        protected String doInBackground(String... voids) {
            String positions = voids[0];
            try{
                Socket soc=new Socket("10.100.9.222",12343);
                Log.d("CONNECTTRY","on essaye de co"+positions);
                //Socket soc=new Socket("192.168.0.13",12343);
                if(soc.isConnected()){

                    Log.d("CONNECT","connecté au socket");

                }
                    /*pw = new PrintWriter(soc.getOutputStream());


                    pw.write(positions);
                    pw.flush();
                    pw.close();
                    BufferedReader in = new BufferedReader(new InputStreamReader(soc.getInputStream()));
                    String reponse = in.readLine();
                    Log.d("REP",reponse);*/

                        //DataOutputStream dout=new DataOutputStream(soc.getOutputStream());
                        //DataInputStream din=new DataInputStream(soc.getInputStream());

                pw = new PrintWriter(soc.getOutputStream());


                pw.write(positions);
                pw.flush();


                BufferedReader in = new BufferedReader(new InputStreamReader(soc.getInputStream(),"UTF-8"));
                response = in.readLine();
                /*for (response= in.readLine(); response != null; response = in.readLine()) {
                    Log.d("MSG_READLINE",response);
                }*/


                /*dout.writeUTF(positions);
                dout.flush();*/


                //str_response = din.readUTF();//in.readLine();




                //dout.close();
                //din.close();
                pw.close();
                in.close();
                soc.close();



            }
            catch(Exception e)
            {
                e.printStackTrace();
            }
            return response;

        }

        @Override
        protected void onPostExecute(String msg_python) {
            super.onPostExecute(msg_python);
            TextView txt = findViewById(R.id.textInfoex);
            if(msg_python != null){
                Log.d("MSG_ONPOST_SOUS_CLASSE",msg_python);
                textInfoex.setText(msg_python);
            }else{
                Log.d("MSG_ONPOST_SOUS_CLASSE","message is null");
            }
        }
    }
}