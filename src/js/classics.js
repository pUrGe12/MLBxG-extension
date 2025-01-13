document.getElementById('load-models').addEventListener('click', async () => {
    const loadButton = document.getElementById('load-models');
    const statusDiv = document.getElementById('load-status');
    
    // Disable button while loading
    loadButton.disabled = true;
    loadButton.style.opacity = '0.7';
    loadButton.textContent = 'Loading Models...';
    
    // Show loading status
    statusDiv.textContent = "Loading model, please wait...";
    statusDiv.style.display = 'block';
    statusDiv.className = 'upload-status success';
    
    try {
        const response = await fetch(`${BASE_URL}/load-yolo-model/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        statusDiv.textContent = "Model loaded successfully!";
        statusDiv.className = 'upload-status success';
        
        loadButton.disabled = false;
        loadButton.style.opacity = '1';
        loadButton.textContent = 'Load Models';
        
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 3000);
        
    } catch (error) {
        console.error('Fetch error:', error);
        statusDiv.textContent = 'Error loading models. Please try again.';
        statusDiv.className = 'upload-status error';
        
        loadButton.disabled = false;
        loadButton.style.opacity = '1';
        loadButton.textContent = 'Load Models';
    }
});

// const BASE_URL = 'https://mlbxg-extension-1.onrender.com';
const BASE_URL = 'http://127.0.0.1:5000';

// Handle match name submission
document.getElementById('name-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const matchName = document.getElementById('match-name').value;
    const responseSection = document.querySelector('.response-section');
    const responseContent = document.querySelector('.response-content');
    const statusDiv = document.getElementById('name-status');
    
    if (!matchName.trim()) {
        statusDiv.textContent = "Please enter a match name";    
        statusDiv.style.display = 'block';
        statusDiv.className = 'upload-status error';
        return;
    }

    // Show processing status
    statusDiv.textContent = "Processing the match, this may take some time...";
    statusDiv.style.display = 'block';
    statusDiv.className = 'upload-status success';
    
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
        
        // Show success status
        statusDiv.textContent = "Match processed successfully!";
        statusDiv.className = 'upload-status success';
        
        // Show response section and content
        responseSection.style.display = 'block';
        responseContent.innerHTML = marked.parse(data.message);
        
        // Scroll to response
        responseSection.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        console.error('Fetch error:', error);
        statusDiv.textContent = 'Error connecting to server. Please try again.';
        statusDiv.className = 'upload-status error';
    }
});

// Handle video submission
document.getElementById('video-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const videoFile = document.getElementById('match-video').files[0];
    const responseSection = document.querySelector('.response-section');
    const responseContent = document.querySelector('.response-content');
    const statusDiv = document.getElementById('video-status');
    
    if (!videoFile) {
        statusDiv.textContent = "Please select a video file";
        statusDiv.style.display = 'block';
        statusDiv.className = 'upload-status error';
        return;
    }

    // Show uploading status
    statusDiv.textContent = "Uploading video...";
    statusDiv.style.display = 'block';
    statusDiv.className = 'upload-status success';
    
    try {
        const formData = new FormData();
        formData.append('video', videoFile);
        const response = await fetch(`${BASE_URL}/classics-video/`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Show success status
        statusDiv.textContent = "Video uploaded successfully!";
        statusDiv.className = 'upload-status success';
        
        // Show response section and content
        responseSection.style.display = 'block';
        responseContent.innerHTML = marked.parse(data.output);
        
        // Scroll to response
        responseSection.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        console.error('Fetch error:', error);
        statusDiv.textContent = 'Error uploading video. Please try again.';
        statusDiv.className = 'upload-status error';
    }
});