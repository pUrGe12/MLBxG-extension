// const BASE_URL = 'https://mlbxg-extension-1.onrender.com';
const BASE_URL = 'http://127.0.0.1:5000';

document.getElementById("submit").addEventListener("click", async () => {
    const userInput = document.getElementById("prompt").value;
    const responseDiv = document.getElementById("output");

    if (!userInput.trim()) {
        responseDiv.textContent = "Please enter something.";
        return;
    }
    else {
        responseDiv.textContent = "Generating response ...";
    }
    

    try {
        const response = await fetch(`${BASE_URL}/process_input`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ input: userInput }),
        });

        if (!response.ok) {
            throw new Error("Failed to connect to the backend.");
        }

        const data = await response.json();
        responseDiv.textContent = data.response || "No response from backend.";
    } catch (error) {
        console.error(error);
        responseDiv.textContent = "An error occurred. Please try again.";
    }
});
