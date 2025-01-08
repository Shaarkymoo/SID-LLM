
#this code gets images using beautiful soup
import time
import requests
from bs4 import BeautifulSoup

def all_engine_search(query):
    google_url = f"https://www.google.com/search?q={query}&source=lnms&tbm=isch"
    bing_url = f"https://www.bing.com/images/search?q={query}"
    yahoo_url = f"https://images.search.yahoo.com/search/images?p={query}"
    ddg_url = f"https://duckduckgo.com/?q={query}&iax=images&ia=images"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    
    google_response = requests.get(google_url, headers=headers)
    bing_response = requests.get(bing_url, headers=headers)
    yahoo_response = requests.get(yahoo_url, headers=headers)
    ddg_response = requests.get(ddg_url, headers=headers)
    
    google_soup = BeautifulSoup(google_response.text, 'html.parser')
    bing_soup = BeautifulSoup(bing_response.text, 'html.parser')
    yahoo_soup = BeautifulSoup(yahoo_response.text, 'html.parser')
    ddg_soup = BeautifulSoup(ddg_response.text, 'html.parser')

    image_elements = google_soup.find_all('img') + bing_soup.find_all('img') + yahoo_soup.find_all('img') + ddg_soup.find_all('img')
    image_urls = []

    for image in image_elements:
        img_url = image.get('data-src') or image.get('src')
        if img_url and 'http' in img_url:
            image_urls.append(img_url)

    return image_urls

#query = "water bottle"
#urls = all_engine_search(query)
#print(urls)
#print(len(urls))


#the part below gets images using selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
import time

def scroll_down(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

def collect_image_links(driver, num_images):
    start = time.time()
    image_urls = set()
    while len(image_urls) < num_images:
        print(len(image_urls))
        scroll_down(driver)
        image_elements = driver.find_elements(By.TAG_NAME, 'img')
        for image in image_elements:
            img_url = image.get_attribute('src') or image.get_attribute('data-src')
            if img_url and 'http' in img_url:
                #print(img_url)
                image_urls.add(img_url)
                if len(image_urls) >= num_images:
                    break
        #print("check1")
        # Check if the page has a "Next" button and click it (for engines like Bing and Yahoo)
        try:
            next_button = driver.find_element(By.XPATH, "//a[contains(@class, 'next')]")
            next_button.click()
            time.sleep(2)
        except:
            pass
        try:
            see_more_button = driver.find_element(By.XPATH, "//a[contains(@class, 'btn_seemore')]")
            see_more_button.click()
            time.sleep(2)
        except:
            pass
        try:
            more_images_button = driver.find_element(By.XPATH, "//button[contains(@name='more-res')]")
            if more_images_button:
                #print("mored")
                driver.execute_script("arguments[0].scrollIntoView(true);", more_images_button)
                time.sleep(1)
                more_images_button.click()
                time.sleep(2)
        except:
            pass
        if time.time()-start>=120:
            return list(image_urls)
    return list(image_urls)

def dynamic_image_search(query, search_engine, num_images=500):
    if search_engine == 'google':
        url = f"https://www.google.com/search?q={query}&source=lnms&tbm=isch"
    elif search_engine == 'bing':
        url = f"https://www.bing.com/images/search?q={query}"
    elif search_engine == 'yahoo':
        url = f"https://images.search.yahoo.com/search/images?p={query}"
    elif search_engine == 'duckduckgo':
        url = f"https://duckduckgo.com/?q={query}&iax=images&ia=images"
    else:
        raise ValueError("Unsupported search engine")
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)    

    driver.get(url)
    time.sleep(5)  # Adjust sleep time as needed
    
    image_urls = collect_image_links(driver, num_images)
    
    driver.quit()
    return image_urls

# Example usage
query = "pistols"
num_images = 500  # Number of images to collect from each search engine
search_engines = ["bing", "google", "duckduckgo","yahoo"] #cannot integrate yahoo
all_image_links = []

for engine in search_engines:
    print("started" , engine)
    image_links = dynamic_image_search(query, engine, num_images)
    all_image_links.extend(image_links)
    print(f"Collected {len(image_links)} images from {engine}")

print(f"Total images collected: {len(all_image_links)}")


#the part below handles the downloading
from fastdownload import download_url
count = 1
for a in all_image_links:
    try:
        dest = 'D:\GPCSSI2024\image dataset\pistols\Pistol' + str(count) + '.jpg'
        download_url(a, dest, show_progress=False)
        if count%50==0:
            print(count,"done")
    except Exception as e:
        pass


#the below code makes sure each image is a valid image in terms of size and format
import cv2
import imghdr
import os

data_dir = f"D:\GPCSSI2024\image dataset\pistols\Pistol" 

image_exts = ['jpeg','jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)