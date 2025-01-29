// Content script to capture video and store in a circular buffer

// Buffer configuration
BUFFER_DURATION_SECONDS = 5; // Buffer duration
FRAME_RATE = 30; // Approximate frame rate
BUFFER_SIZE = BUFFER_DURATION_SECONDS * FRAME_RATE; // Total frames in the buffer

buffer = []; // Circular buffer for video frames
bufferIndex = 0; // Current index in the buffer

console.log("I am here!");

// Helper function to initialize video capture
function captureVideo() {
    const videoElement = document.querySelector("video"); // Target the video element

    if (!videoElement) {
        console.error("No video element found!");
        return;
    }
    // Create a canvas to capture frames
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    // Set up canvas dimensions
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    // Start capturing frames
    const captureInterval = setInterval(() => {
        if (videoElement.paused || videoElement.ended) {
            clearInterval(captureInterval);
            return;
        }

        // Draw current video frame to the canvas
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        // Convert the canvas to a Blob and store it in the buffer
        canvas.toBlob((blob) => {
            if (buffer.length < BUFFER_SIZE) {
                buffer.push(blob);
                console.log("Added new frame at index:", buffer.length - 1);
            } else {
                console.log("Overwriting frame at index:", bufferIndex);
                buffer[bufferIndex] = blob;
                bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
            }
        }, "image/webp"); // Use a lightweight format
    }, 1000 / FRAME_RATE); // Capture at the specified frame rate
}

// captureVideo();
// Listen for video playback start
document.addEventListener("play", (event) => {
    if (event.target.tagName === "VIDEO") {
        console.log("Video playback started. Initializing capture...");
        captureVideo();
    }
}, true);

// Provide a way to retrieve the buffer
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "getBuffer") {
        // Combine buffer frames into a video Blob
        const videoBlob = new Blob(buffer, { type: "video/webm" });
        sendResponse({ videoBlob });
    }
});
