const BASE_URL = 'https://mlbxg-extension-1.onrender.com';

// Helper function to show response
function showResponse(message) {
    const responseSection = document.querySelector('.response-section');
    const responseContent = document.querySelector('.response-content');
    responseSection.style.display = 'block';
    responseContent.innerHTML = marked.parse(message);
    
    // Scroll to response
    responseSection.scrollIntoView({ behavior: 'smooth' });
}

// Helper function to show status messages
function showStatus(statusDiv, message, isSuccess) {
    statusDiv.textContent = message;
    statusDiv.style.display = 'block';
    statusDiv.className = `upload-status ${isSuccess ? 'success' : 'error'}`;
}

// Handle match name submission
document.getElementById('name-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const matchName = document.getElementById('match-name').value;
    const statusDiv = document.getElementById('name-status');
    
    if (!matchName.trim()) {
        showStatus(statusDiv, "Please enter a match name", false);
        return;
    }

    showStatus(statusDiv, "Processing match name...", true);
    
    try {
        const response = await fetch(`${BASE_URL}/classics-text/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input: matchName }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        showStatus(statusDiv, "Match name submitted successfully!", true);
        
        // Show the response from the backend
        if (data.response) {
            showResponse(data.response);
        }
    } catch (error) {
        console.error('Fetch error:', error);
        showStatus(statusDiv, "Error connecting to server. Please try again.", false);
    }
});

// Handle video submission
document.getElementById('video-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const videoFile = document.getElementById('match-video').files[0];
    const statusDiv = document.getElementById('video-status');
    
    if (!videoFile) {
        showStatus(statusDiv, "Please select a video file", false);
        return;
    }

    showStatus(statusDiv, "Uploading video...", true);
    
    try {
        const formData = new FormData();
        formData.append('video', videoFile);

        const response = await fetch(`${BASE_URL}/classics-video/`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        showStatus(statusDiv, "Video uploaded successfully!", true);
        
        // Show the response from the backend
        if (data.response) {
            showResponse(data.response);
        }
    } catch (error) {
        console.error('Fetch error:', error);
        showStatus(statusDiv, "Error uploading video. Please try again.", false);
    }
});