// Content script for video capture
const BUFFER_DURATION_SECONDS = 5;
const FRAME_RATE = 30;
const BUFFER_SIZE = BUFFER_DURATION_SECONDS * FRAME_RATE;
let mediaRecorder = null;
let recordedChunks = [];
let isBufferSending = false;

function captureVideo() {
    const videoElement = document.querySelector("video");
    if (!videoElement) {
        console.error("No video element found!");
        return;
    }

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const stream = canvas.captureStream(FRAME_RATE);
    mediaRecorder = new MediaRecorder(stream, {
        mimeType: "video/webm;codecs=vp9"
    });

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
            if (recordedChunks.length > BUFFER_SIZE) {
                recordedChunks.shift();
            }
            // Try to send buffer after each chunk
            trySendBuffer();
        }
    };

    mediaRecorder.start(1000 / FRAME_RATE);

    const captureInterval = setInterval(() => {
        if (videoElement.paused || videoElement.ended) {
            clearInterval(captureInterval);
            return;
        }
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    }, 1000 / FRAME_RATE);
}

async function trySendBuffer() {
    if (isBufferSending || recordedChunks.length < BUFFER_SIZE) {
        return;
    }
    
    isBufferSending = true;
    
    try {
        const videoBlob = new Blob(recordedChunks, { type: "video/webm" });
        const formData = new FormData();
        formData.append("video", videoBlob, "capture.webm");

        await fetch("http://localhost:5000/check-buffer", {
            method: "POST",
            body: formData
        });
    } catch (error) {
        console.log("Buffer not needed yet");
    } finally {
        isBufferSending = false;
    }
}

document.addEventListener("play", (event) => {
    if (event.target.tagName === "VIDEO") {
        captureVideo();
    }
}, true);

// Listen for messages from popup/background
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "processRequest") {
        // Send the query to process_input endpoint
        fetch("http://localhost:5000/process_input", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ input: message.query })
        })
        .then(response => response.json())
        .then(result => sendResponse(result))
        .catch(error => sendResponse({error: error.message}));
        return true;
    }
});
