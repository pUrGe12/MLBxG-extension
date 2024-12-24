document.getElementById('stats-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const stats = document.getElementById('stats-input').value;
            const resultDiv = document.getElementById('prediction-result');
            const predictionText = document.getElementById('prediction-text');

            try {
                const response = await fetch('http://127.0.0.1:5000/user-stat/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ stats })
                });

                const data = await response.json();
                predictionText.textContent = data.prediction;
                resultDiv.style.display = 'block';
            } catch (error) {
                predictionText.textContent = 'Error connecting to server. Please try again.';
                resultDiv.style.display = 'block';
            }
        });