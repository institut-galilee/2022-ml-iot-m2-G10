package com.example.examonitor;

import android.os.AsyncTask;
import android.util.Log;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class ConnectServer extends AsyncTask<String,Void,String> {
    Socket soc;
    PrintWriter pw;
    String response;

    @Override
    protected String doInBackground(String... voids) {
        String positions = voids[0];
        try{
            Log.d("CONNECTTRY","on essaye de co1");
            //Socket soc=new Socket("10.0.2.2",12343);
            Socket soc=new Socket("192.168.137.1",12343);
            if(soc.isConnected()){
                CharSequence text = "La connexion avec le système a été réalisée";
                //Context context = GlobalApplication.getAppContext();
                Log.d("CONNECT","connecté au socket");

            }
            pw = new PrintWriter(soc.getOutputStream());
            pw.write(positions);
            pw.flush();
            BufferedReader in = new BufferedReader(new InputStreamReader(soc.getInputStream(),"UTF-8"));

            /*for (response= in.readLine(); response != null; response = in.readLine()) {
                Log.d("MSG_READLINE",response);
            }*/
            //response = in.readLine();
            /*dout.writeUTF(positions);
            dout.flush();*/


            //str_response = din.readUTF();//in.readLine();




            //dout.close();
            //din.close();
            pw.close();
            in.close();
            //soc.close();



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
        //TextView txt = findViewById(R.id.textInfoac);
        if(msg_python != null){
            Log.d("MSG_ONPOST",msg_python);
            //txt_acc.setText(msg_python);
        }
    }
}