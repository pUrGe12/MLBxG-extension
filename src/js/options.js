document.getElementById('stats-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const stats = document.getElementById('stats-input').value;
    const responseDiv = document.getElementById('prediction-text');
    const predictionResultDiv = document.getElementById('prediction-result');

    if (!stats.trim()) {
        predictionResultDiv.style.display = 'block';
        responseDiv.innerHTML = "Please enter something...";
        return;
    }

    predictionResultDiv.style.display = 'block';
    responseDiv.innerHTML = "Generating prediction!...";

    try {
        const response = await fetch('http://127.0.0.1:5000/user-stat/', {
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
        
        // Parse markdown to HTML using marked
        const htmlContent = marked.parse(data.response, {
            breaks: true, // Enable line breaks
            gfm: true,    // Enable GitHub Flavored Markdown
        });
        
        // Set the parsed HTML content
        responseDiv.innerHTML = htmlContent;
    } catch (error) {
        console.error('Fetch error:', error);
        responseDiv.innerHTML = 'Error connecting to server. Please try again.';
    }
});