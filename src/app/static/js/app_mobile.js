const video = document.getElementById('video');
const replay = document.getElementById('replay');
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const replayButton = document.getElementById('replayBtn');
const approveButton = document.getElementById('approve');
const timerElement = document.getElementById('timer');
const resultElement = document.getElementById('result');
const resultDiv = document.getElementById("result");
const optionsDiv = document.getElementById('options');

let stream;
let interval;
let recognizedSigns = [];
let recognizedSentence = [];
let optionsList = [];

startButton.addEventListener("click", async () => {
    startButton.disabled = true;
    stopButton.disabled = false;


    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false});
    video.srcObject = stream;
    // console.log("here")
    startActivate();
    interval = setInterval(captureAndSendFrame, 200); // Send a frame every 1000ms (1 second)
    // TODO for Kenneth, maybe send images every 100 ms then server will just wait until 1 second
    // this will give server 10 images to work with.
});

stopButton.addEventListener("click", () => {
    startButton.disabled = false;
    stopButton.disabled = true;

    clearInterval(interval);
    video.srcObject.getTracks().forEach(track => track.stop());
});

function startActivate() {

    try {
        const response = fetch("/api/recognize", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ image:'', activate: 1 })
        });
        // resultDiv.innerHTML = `<p>a</p>`;
    } catch (error) {
        console.error("Error sending frame to the server:", error);
    }
}

async function captureAndSendFrame() {
    const canvas = document.createElement("canvas");
    console.log(video.videoWidth, video.videoHeight)
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL("image/jpeg", 0.8).split(",")[1];

    try {
        const response = await fetch("/api/recognize", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ image: imageData, activate: 0  })
        });

        let result = await response.json();
        
        console.log(result.sign);
        console.log(result.sentence);
        console.log(typeof result.sign);
        console.log(result.options); //this contains the array of options
        
        if(result.sign.length > 0){
            recognizedSigns.push(result.sign);
            recognizedSentence.push(result.sentence);
            optionsList = result.options;
            updateResultDisplay();
            updateOptions();
        } 
        // resultDiv.innerHTML = `<p>a</p>`;
    } catch (error) {
        console.error("Error sending frame to the server:", error);
    }
}

function updateResultDisplay() {
    /* 
        * This function updates the current results that we have ... we'll modify to have a sentence and
        * not bullet points ... 
        */
    resultDiv.innerHTML = "<p>";
    recognizedSentence.forEach(sentence => {
        resultDiv.innerHTML = `${sentence}`;
    });
    resultDiv.innerHTML += "</p>";
    
}

function updateOptions() {
    /* 
        * This function updates the current results that we have ... we'll modify to have a sentence and
        * not bullet points ... 
        */
    recognizedSigns.forEach(sign => {
        optionsDiv.innerHTML = `<br><b class="text-center card-header"> Word: ${[sign]}<b>`;
    });

    optionsDiv.innerHTML += `<br><br>`

    var ol = document.createElement("ol");

    // ol.className = 'list-group-numbered'
    optionsDiv.appendChild(ol);
    optionsList.forEach(options => {
        var li = document.createElement("li");
        // li.className = 'list-group-item'
        li.innerHTML = `${options}`;
        ol.appendChild(li);
    });
    optionsDiv.appendChild(ol);
    
}

// let recordedChunks = [];
// let mediaRecorder;

// navigator.mediaDevices.getUserMedia({ video: true })
//     .then(stream => {
//         video.srcObject = stream;
//     });

// startButton.addEventListener('click', async () => {
//     timerElement.textContent = '3';
//     startButton.style.display = 'none';
//     await sleep(1000);
//     timerElement.textContent = '2';
//     await sleep(1000);
//     timerElement.textContent = '1';
//     await sleep(1000);
//     timerElement.textContent = '';

//     startRecording();
// });

// stopButton.addEventListener('click', () => {
//     stopRecording();
// });

// replayButton.addEventListener('click', () => {
//     replay.play();
// });

// approveButton.addEventListener('click', async () => {
//     replay.pause();
//     replay.currentTime = 0;
//     const blob = new Blob(recordedChunks, { type: 'video/webm' });

//     const videoData = await blobToArrayBuffer(blob);
//     const base64Data = arrayBufferToBase64(videoData);

//     const response = await fetch('/recognize', {
//         method: 'POST',
//         body: JSON.stringify({ video: base64Data }),
//         headers: { 'Content-Type': 'application/json' }
//     });


//     if (response.ok) {
//         const { result } = await response.json();
//         resultElement.textContent = result;
//     } else {
//         resultElement.textContent = 'Error recognizing ASL.';
//     }

//     resetUI();
// });

// function sleep(ms) {
//     return new Promise(resolve => setTimeout(resolve, ms));
// }

// function startRecording() {
//     navigator.mediaDevices.getUserMedia({ video: true, audio: false })
//         .then(stream => {
//             mediaRecorder = new MediaRecorder(stream);
//             mediaRecorder.ondataavailable = (event) => {
//                 if (event.data.size > 0) {
//                     recordedChunks.push(event.data);
//                 }
//             };
//             mediaRecorder.start();
//             stopButton.style.display = 'inline-block';
//         });
// }

// function stopRecording() {
//     mediaRecorder.stop();
//     stopButton.style.display = 'none';
//     replayButton.style.display = 'inline-block';
//     approveButton.style.display = 'inline-block';
//     video.style.display = 'none';
//     replay.style.display = 'inline-block';

//     setTimeout(() => {
//         const blob = new Blob(recordedChunks, { type: 'video/webm' });
//         replay.src = URL.createObjectURL(blob);
//     }, 100);
// }

// function resetUI() {
//     video.style.display = 'inline-block';
//     replay.style.display = 'none';
//     startButton.style.display = 'inline-block';
//     replayButton.style.display = 'none';
//     approveButton.style.display = 'none';
//     recordedChunks = [];
// }

// function blobToArrayBuffer(blob) {
//     return new Promise((resolve, reject) => {
//         const reader = new FileReader();
//         reader.onload = () => {
//             resolve(reader.result);
//         };
//         reader.onerror = () => {
//             reject(new Error('Error reading Blob as ArrayBuffer'));
//         };
//         reader.readAsArrayBuffer(blob);
//     });
// }

// function arrayBufferToBase64(buffer) {
//     const binary = [];
//     const bytes = new Uint8Array(buffer);
//     const len = bytes.byteLength;

//     for (let i = 0; i < len; i++) {
//         binary.push(String.fromCharCode(bytes[i]));
//     }

//     return btoa(binary.join(''));
// }
