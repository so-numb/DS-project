import pandas as pd
import requests
from bs4 import BeautifulSoup


product_name = []
product_price = []
product_description = []
product_rating = []
product_review = []


for i in range(1, 11):
    n = 1
    if

#     np = soup.find("a", class_ = "cn++Ap A1msZJ").get("href")
#     print("https://www.flipkart.com"+np)


url = "https://www.flipkart.com/search?q=mobiles+under+50000&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&page=1"

headers = {
        "authority": "www.google.com",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "max-age=0",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        # add more headers as needed
    }

# set headers
response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, "lxml")
box = soup.find("div", class_ = "DOjaWF gdgoEp")

names = box.find_all("div", class_="KzDlHZ")
for i in names:
    name = i.text
    product_name.append(name)
print(product_name)
print(len(product_name))

prices = box.find_all("div", class_ ="Nx9bqj _4b5DiR")
for i in prices:
    price = i.text
    product_price.append(price)
print(product_price)
print(len(product_price))

descriptions = box.find_all("ul", class_ = "G4BRas")
for i in descriptions:
    description = i.text
    product_description.append(description)
print(product_description)
print(len(product_description))

ratings = box.find_all("div", class_ = "XQDdHH")
for i in ratings:
    rating = i.text
    product_rating.append(rating)
print(product_rating)
print(len(product_rating))

rating_reviews = box.find_all("span", class_ = "Wphh3N")
for i in rating_reviews:
    review1 = soup.find_all("span")
    review = i.text
    product_review.append(review)
print(product_review)
print(len(product_review))

# df = pd.DataFrame({"Product Name":product_name, "Price":product_price, "Description":product_description, "Ratings":product_rating, "Review":product_review})
# print(df)
