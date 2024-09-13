"""Create function that takes in URL and fetches:

text (before paywall) (BS4)

text after paywall (what happens if there is not paywall?) (BS4)

number of comments (BS4)

date/time (BS4)

title (newspaper)

author (newspaper)

if the article is behind the paywall or not (BS4)"""

from bs4 import BeautifulSoup
from newspaper import Article
import datetime
# %%

def get_current_date():
    # Get the current date
    today = datetime.date.today()
    # Format the date as DD MM YYYY
    return today.strftime('%d%m%Y')

def convert_ausgabe_string_to_date(week_year_str:str):
    # Remove the 'Ausgabe ' prefix and split the input string into week and year
    prefix = 'Ausgabe '
    if week_year_str.startswith(prefix):
        week_year_str = week_year_str[len(prefix):]

    week_str, year_str = week_year_str.split('/')
    week = int(week_str)
    year = int(year_str)
    
    # Calculate the Monday of the week (using isocalendar)
    first_day_of_year = datetime.date(year, 1, 1)
    # If the first day of the year is not Monday, adjust to the first Monday
    first_monday_of_week = first_day_of_year + datetime.timedelta(days=(week - 1) * 7)
    while first_monday_of_week.isocalendar()[2] != 1:  # 1 is Monday
        first_monday_of_week += datetime.timedelta(days=1)
    
    # Calculate the Thursday of that week
    thursday_of_week = first_monday_of_week + datetime.timedelta(days=3)
    
    # Return the date formatted as DD MM YYYY
    return thursday_of_week.strftime('%d %m %Y')

def retrieve_date_paywall_text(article_html):
    soup = BeautifulSoup(article_html, 'html.parser')

    # TEXT AND PAYWALL
    # only for non-paywall articles
    text_class = "column s-article-text js-dynamic-advertorial js-external-links"
    text = (soup
            .find("div",
                  {"class": text_class}))
    # get paywall-article introduction paragraph
    intro_class = "column s-article-text c-paywall-hidden-text js-dynamic-advertorial js-external-links"
    paywall_intro = (soup
                     .find("div",
                           {"class": intro_class}))
    
    # get content behind paywall
    paywall_class= "o-paywall"
    paywall_text = (soup
                    .find('div',
                          {'class': paywall_class}))
    paywall = False
    
    # check what has been collected
    if text:
        text = text.get_text()
    elif paywall_intro and paywall_text:
        paywall = True
        paywall_intro = paywall_intro.get_text()
        paywall_text = paywall_text.get_text()[12:] # first chars are "\n         "

        def combine_strings(str1, str2):
            # Find the longest suffix of str1 that matches the prefix of str2
            overlap_len = 0
            for i in range(1, len(str1) + 1):
                if str2.startswith(str1[-i:]):
                    overlap_len = i
            
            # Combine the strings by removing the overlapping part from str2
            combined_string = str1 + str2[overlap_len:]
            return combined_string
        text = combine_strings(paywall_intro, paywall_text)

    # DATE
    date = soup.find("span", class_="js-article-issue-name")
    if date:
        date = date.get_text()
        date = convert_ausgabe_string_to_date(date)
    else:
        date = get_current_date()

    return date, paywall, text

def fetch_article(article_url):
    article = Article(article_url)
    article.download()
    article.parse()
    authors = article.authors
    title: str = article.title
    date, paywall, text = retrieve_date_paywall_text(article.html)
    return [article_url, title, authors, date, paywall, text]
# %%
