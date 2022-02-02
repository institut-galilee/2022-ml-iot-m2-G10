var MediaStreamd;
        var strm;
        function desactVideo(){
            MediaStreamd.stop();
            
            document.querySelector("#videoInput").srcObject=null;
            if (document.querySelector('#btn_cam').innerText =='Désactiver ma vidéo' ){
                document.querySelector('#btn_cam').innerText = 'Activer ma vidéo';
                document.getElementById("btn_cam").setAttribute( "onClick", "javascript: demarrerVideo();" );
            };
            document.getElementById("btn_cam").setAttribute( "onClick", "javascript: demarrerVideo();" );
        }
        function demarrerVideo(){
            if (navigator.mediaDevices === undefined) {
                navigator.mediaDevices = {};
            }
            console.log('On démarre la vidéo.');
            if (document.querySelector('#btn_cam').innerText =='Activer ma vidéo' ){
                document.querySelector('#btn_cam').innerText = 'Désactiver ma vidéo';
                document.getElementById("btn_cam").setAttribute( "onClick", "javascript: desactVideo();" );
            };
            let video = document.querySelector("#videoInput");
            console.log(video.srcObject);
            if(navigator.mediaDevices.getUserMedia){

                navigator.mediaDevices.getUserMedia({
                    video:true
                }).then(function(stream){
                    strm = stream;
                    video.srcObject = stream;
                    MediaStreamd = stream.getTracks()[0];
                })
                .catch(function(error){
                    console.log("Erreur caméra");
                })
            } else {
                console.log('GetUserMedia nest pas supporté.');
            }
        }