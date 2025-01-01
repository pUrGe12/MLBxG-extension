chrome.action.onClicked.addListener(async (tab) => {
  // Open the side panel
  await chrome.sidePanel.open({tabId: tab.id});
});

chrome.webNavigation.onCompleted.addListener((details) => {
  if (details.url && details.url.includes("youtube.com/watch")) {
    chrome.scripting.executeScript({
      target: { tabId: details.tabId },
      files: ["content.js"]
    });
  }
}, { url: [{ urlMatches: 'youtube.com/watch' }] });

// added functionality to send out the buffer 

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "getBuffer") {
        // Forward the message to the content script
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const activeTab = tabs[0];
            chrome.tabs.sendMessage(activeTab.id, { action: "getBuffer" }, (response) => {
                if (response) {
                    sendResponse(response);
                } else {
                    sendResponse({ error: "Failed to get buffer" });
                }
            });
        });
        return true; // Keeps the message channel open for async responses
    }
});