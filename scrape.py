# %%
from newspaper import build
import pandas as pd

from freitag import fetch_article
from freitag import get_current_date


freitag = build("https://www.freitag.de", language="de", memorize_articles=False)
article_urls = freitag.article_urls()

df = (pd.DataFrame(data={"url":[], "title":[], "authors":[], "date":[],
                        "paywall":[], "text":[]})
    .set_index("url"))

for i, article_url in enumerate(article_urls):
    print(f"---- Collecting Article #{i+1} ----")
    row = pd.Series(fetch_article(article_url),
                    index=["url", "title", "authors", "date", "paywall", "text"])
    if article_url not in df.index:
        df.loc[article_url] = row
    else:
        print(f"---- Article {i+1} already stored ----")

df.to_csv(f"./data/freitag_{get_current_date()}.csv")


# %%
