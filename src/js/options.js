document.getElementById('stats-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const stats = document.getElementById('stats-input').value;
    const responseDiv = document.getElementById('prediction-text');
    const predictionResultDiv = document.getElementById('prediction-result');

    if (!stats.trim()) {
        // Show error message immediately
        predictionResultDiv.style.display = 'block';
        responseDiv.textContent = "Please enter something...";
        return;
    }

    // Show "Generating prediction!..." immediately
    predictionResultDiv.style.display = 'block';
    responseDiv.textContent = "Generating prediction!...";

    try {
        const response = await fetch('http://127.0.0.1:5001/user-stat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input: stats }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        responseDiv.textContent = data.response;
    } catch (error) {
        console.error('Fetch error:', error);
        responseDiv.textContent = 'Error connecting to server. Please try again.';
    }
});