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
