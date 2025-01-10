from selenium import webdriver
from selenium.webdriver.common.by import By
import time


def get_video_results():
    driver = webdriver.Chrome()
    driver.get('https://www.youtube.com/results?search_query=minecraft')

    youtube_data = []

    # scrolling to the end of the page
    # https://stackoverflow.com/a/57076690/15164646
    while True:
        try:
            # end_result = "No more results" string at the bottom of the page
            # this will be used to break out of the while loop
            end_result = driver.find_element(By.CSS_SELECTOR, '#video-title > yt-formatted-string').is_displayed()
            # time.sleep(1) # could be removed

            # once the element is located, break out of the loop
            if end_result:
                break
        except Exception as e:
            driver.execute_script("var scrollingElement = (document.scrollingElement || document.body);scrollingElement.scrollTop = scrollingElement.scrollHeight;")

    print('Extracting results. It might take a while...')

    # iterate over all elements and extract link
    for result in driver.find_elements(By.CSS_SELECTOR, '.text-wrapper.style-scope.ytd-video-renderer'):
        link = [i.text for i in result.find_elements(By.CSS_SELECTOR, '.title-and-badge.style-scope.ytd-video-renderer a')]
        youtube_data.append(link)

    return youtube_data

print(get_video_results())

# prints all found links