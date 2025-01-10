import os
from yt_dlp import YoutubeDL

from selenium import webdriver
from selenium.webdriver.common.by import By
import time

eric_clapton_url = "https://www.youtube.com/watch?v=3U4yDkvRjvs"
three_blue_url = "https://www.youtube.com/watch?v=piJkuavhV50"

user_query = "1975 World Series Game 6 where the Boston Red Sox beat the Cincinnati Reds"				# we can use an LLM to get the exact search parameter too

def get_url(user_query):
	driver = webdriver.Chrome()
	driver.get(f'https://www.youtube.com/results?search_query={user_query}')

	# This will always find the first link of the video (it won't pick up the sponsor messages so chill)
	link = driver.find_element(By.XPATH, "/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[1]/div[1]/div/div[1]/div/h3/a").get_attribute('href')

	# link has currently the & part which is something we don't care about
	link = str(link.split('&')[0])

	return link

custom_url = get_url(user_query)													# based on user's input search youtube and select the first link

path = '/home/purge/Desktop/MLBxG-extension/test/youtube_scraping/downloads'

ydl_opts = {
    'outtmpl': os.path.join(path, '%(title)s.%(ext)s'),
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([custom_url])
