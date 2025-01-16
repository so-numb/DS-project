import pandas as pd
import requests
from bs4 import BeautifulSoup
import re


headers = {
        "authority": "www.google.com",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "max-age=0",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    }


price_list = []
ratings_list = []
titles_list = []
reviews_list = []
phone_list = []

for i in range (1, 207):
    print(i)
    # URL for the current page
    url = "https://www.flipkart.com/apple-iphone-15-plus-black-128-gb/product-reviews/itme3a53984760fb?pid=MOBGTAGPNRQA7CS3&lid=LSTMOBGTAGPNRQA7CS3YEKCQO&marketplace=FLIPKART&page=" + str(i)

    # Sending a GET request to get the HTML content
    page = requests.get(url, headers=headers)

    # Parsing the HTML content
    soup = BeautifulSoup(page.content, 'html.parser')

    # Extracting phone name
    phoneName = soup.find_all('div', class_='Vu3-9u eCtPz5')
    for pName in phoneName:
        phone_list.append(pName.get_text())

    # Extracting price
    price = soup.find_all('div', class_='Nx9bqj')
    for p in price:
        price_list.append(p.get_text())

    # Extracting ratings
    rating = soup.find_all('div', class_='XQDdHH Ga3i8K')
    for r in rating:
        rating_star = r.get_text()
        if rating_star:
            ratings_list.append(rating_star)
        else:
            ratings_list.append('0')  # Replace null ratings with 0

    # Extracting review titles
    title = soup.find_all('p', class_='z9E0IG')
    for t in title:
        titles_list.append(t.get_text())

    # Extracting reviews
    review = soup.find_all('div', class_='ZmyHeo')
    for c in review:
        review_text = c.div.div.get_text(strip=True)
        reviews_list.append(review_text)

# Ensuring same length for all the lists
min_length = min(len(phone_list), len(price_list), len(titles_list), len(ratings_list), len(reviews_list))
phone_list = phone_list[:min_length]
print(len(phone_list))
price_list = price_list[:min_length]
print(len(price_list))
ratings_list = ratings_list[:min_length]
print(len(ratings_list))
titles_list = titles_list[:min_length]
print(len(titles_list))
reviews_list = reviews_list[:min_length]
print(len(reviews_list))

# Creating a DataFrame from the data stored in the lists
data = {
    'Phone': phone_list,
    'Price': price_list,
    'Rating': ratings_list,
    'Review Title': titles_list,
    'Review': reviews_list
}

# loading data into a DataFrame object
df_new = pd.DataFrame(data)

# append data frame to CSV file
df_new.to_csv('Apple_iPhone_review.csv', mode='a', index=False, header=False)