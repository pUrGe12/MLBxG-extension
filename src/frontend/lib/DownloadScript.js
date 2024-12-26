const https = require('https');
const fs = require('fs');
const path = require('path');

const url = 'https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js';
const libDir = path.join(__dirname, 'lib');
const filePath = path.join(libDir, 'marked.min.js');

// Create lib directory if it doesn't exist
if (!fs.existsSync(libDir)) {
    fs.mkdirSync(libDir);
}

// Download the file
https.get(url, (response) => {
    const fileStream = fs.createWriteStream(filePath);
    response.pipe(fileStream);

    fileStream.on('finish', () => {
        console.log('marked.js downloaded successfully');
        fileStream.close();
    });
}).on('error', (err) => {
    console.error('Error downloading marked.js:', err);
});