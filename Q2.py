import requests
from bs4 import BeautifulSoup
import pandas as pd
from operator import itemgetter
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt

# Defining events and keywords to be used for sentiment analysis
events = {
    "Policitcs": ["politics","anc","eff","corruption","goverment","eff","minister","leader","cyril","president","legal","court","magistrate",],
    "SA": ["unemployment", "school","flood","earthquake","loadshedding","president","SA","load shedding","eskom","generator","power","electricity",],
    "Business": ["gupta","finance","trade","stocks","nasper","currency","business",],
    "Technology": ["ai", "tools", "technology", "phone", "elon musk"],
    "Entertainment": ["social", "media", "food", "music", "healthy", "lifestyle"],
}

# Getting the url from News24
url = "https://www.news24.com/"
newsFeed = requests.get(url)
beautSoap = BeautifulSoup(newsFeed.text, "html.parser")

# Extract headlines and dates from the articles
news_data = []
for idx, articleTitle in enumerate(
    beautSoap.find_all("div", class_="article-item__title"), start=1
):
    headline = articleTitle.text.strip()
    date = articleTitle.find_next("p", class_="featured-category__date")
    date = articleTitle.find_next("p", class_="article-item__date")
    date = date.text.strip() if date else None
    news_data.append((headline, date))

# Sort the headlines in descending order
descend_order = sorted(news_data, key=itemgetter(0), reverse=True)

# Creates the dataframe using pandas
df = pd.DataFrame(descend_order, columns=["News Headline", "Date & Time"]).reset_index(
    drop=True
)

# Prints original database
print(df)

# NLTK
nltk = stopwords.words("english")

# TextBlob removes any Stop Words from the columns and creates a new dataframe
stpWord_df = pd.DataFrame(
    { "Stop Words Removed": [" ".join([word.lower()
            for word in TextBlob(headline).words
            if word.lower() not in nltk
    ]).strip()for headline in df["News Headline"]]
    }
)

# Prints the cleaned headlines
print("-------------------------------------------------------------")
print(stpWord_df.reset_index(drop=True))

freqCount = df["News Headline"].str.split(expand=True).stack().value_counts()
mostWords = freqCount.head(100)

plt.figure(figsize=(18, 6))  # Adjust the figure size as per your preference
mostWords.plot(kind="bar", color="cyan", width=0.5)
plt.xlabel("Words", fontweight="bold")
plt.ylabel("Frequency", fontweight="bold")
plt.title("Top 100 Words", fontsize=15, fontweight="bold")
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility

for i, x in enumerate(mostWords):
    plt.text(i, x, str(x), ha="center", va="center", fontweight="bold")

plt.tight_layout()  # Adjust the spacing between the plot elements
plt.show()

# Fetch news headlines and dates from News24
selected_headlines = []
sentiment_polarity_list = []

for headline, _ in news_data:
    selected_headlines.append(headline)
    blob = TextBlob(headline)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_polarity_list.append(sentiment_polarity)
    print("News Headline:", headline)
    print("Sentiment Polarity:", sentiment_polarity)
    print("---")

colors = ["red" if polarity < 0 else "green" for polarity in sentiment_polarity_list]

plt.figure(figsize=(10, 6))
plt.bar(range(len(sentiment_polarity_list)), sentiment_polarity_list, color=colors)
plt.title("Daily Happiness Index")
plt.ylabel("Sentiment Polarity")
plt.xticks([])  # Remove x-axis tick labels
plt.show()

# Sentiment trend analysis for each event
eventSenti = {}
for event, keywords in events.items():
    eventNews = [
        headline
        for headline, _ in news_data
        if any(keyword in headline.lower() for keyword in keywords)
    ]
    sentiment_polarity_list = []
    dates = []
    for headline, date in news_data:
        if headline in eventNews:
            blob = TextBlob(headline)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_polarity_list.append(sentiment_polarity)
            dates.append(date)
    eventSenti[event] = {
        "sentiment_polarity": sentiment_polarity_list,
        "dates": dates,
    }

# Plot sentiment patterns for each event
for event, data in eventSenti.items():
    colours = [
        "green" if polarity >= 0 else "red" for polarity in data["sentiment_polarity"]
    ]
    plt.scatter(data["dates"], data["sentiment_polarity"], color=colours)
    plt.xlabel("Date")
    plt.ylabel("Sentiment Polarity")
    plt.title(f"News Trend Analysis - {event}")
    plt.show()
