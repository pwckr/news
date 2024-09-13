# %%
import os

import pandas as pd

file_list = sorted(os.listdir("data"))

all_dfs= [pd.read_csv("data/"+file) for file in file_list]
for df in all_dfs:
    df.index = df["url"]

df = pd.concat(all_dfs, axis=0)
df = df[~df.index.duplicated(keep='first')]
df = df[["url", "title", "authors", "date", "paywall", "text"]]
# %%
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['text'] = df['text'].fillna('')
df["text"] = df["text"].str.lower()
df["title"] = df["title"].str.lower()
df['text_length'] = df['text'].apply(len)
authors_to_remove = ["['Freitag-Veranstaltungen']", "[]", "['der Freitag Podcast']"]
df = df[~df["authors"].isin(authors_to_remove)]

# %%
import re

# Function to remove special characters
def remove_special_characters(text):
    # Remove special characters, retaining spaces and alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9\säöüßÄÖÜ]', '', text)

# Apply the function to 'title' and 'text' columns
df['cleaned_title'] = df['title'].apply(remove_special_characters)
df['cleaned_text'] = df['text'].apply(remove_special_characters)
# %%



# %%
import spacy

# Load the German spaCy model
nlp = spacy.load('de_core_news_sm')
# %%
# Function to process text using spaCy for tokenization, stop word removal, and lemmatization
def spacy_process_text(text):
    # Process the text
    doc = nlp(text)
    # Generate list of lemmatized tokens after removing stop words and punctuation
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(lemmatized_tokens)

# Apply the spaCy processing to 'cleaned_title' and 'cleaned_text'
df['processed_title'] = df['cleaned_title'].apply(spacy_process_text)
df['processed_text'] = df['cleaned_text'].apply(spacy_process_text)

# Show the updated DataFrame
df[['cleaned_title', 'processed_title', 'cleaned_text', 'processed_text']].head()

# %%
df.to_csv("data/combined_data.csv")


# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'processed_text' column to create TF-IDF features
tfidf_features = tfidf_vectorizer.fit_transform(df['processed_text'])
tfidf_features.shape
# %%
from sklearn.cluster import KMeans

# Initialize KMeans with 5 clusters
k = 6
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the model on the TF-IDF features
kmeans.fit(tfidf_features)

# Retrieve the cluster labels for each document
cluster_labels = kmeans.labels_

# Print the first few labels to get an idea of the cluster assignment
print(cluster_labels[:10])
df["cluster"] = cluster_labels
# %%
df["cluster"].value_counts()
# %%
# Assuming 'tfidf_features' is your TF-IDF matrix and 'cluster_labels' are your K-means cluster assignments

# Step 1: Extract feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Step 2: Calculate mean TF-IDF per cluster
# Create a DataFrame for the TF-IDF features
df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=feature_names)
df_tfidf['cluster'] = cluster_labels

# Initialize a dictionary to hold the top words for each cluster
top_words_per_cluster = {}

# Step 3: Get top 10 words per cluster
for cluster in range(k):  # Adjust 'k' to your number of clusters
    # Filter the DataFrame for a single cluster
    cluster_data = df_tfidf[df_tfidf['cluster'] == cluster]
    
    # Calculate mean TF-IDF score per word for the cluster
    mean_scores = cluster_data.drop('cluster', axis=1).mean(axis=0)
    
    # Sort words based on mean score and get the top 10 words
    top_words = mean_scores.sort_values(ascending=False).head(10).index.tolist()
    
    # Store the top words in the dictionary
    top_words_per_cluster[cluster] = top_words

# Print the top words for each cluster
for cluster, words in top_words_per_cluster.items():
    print(f'Cluster {cluster}: {words}')

# %%
# Assuming you have a dictionary `top_words_per_cluster` from previous steps
for cluster, words in top_words_per_cluster.items():
    heading = ' | '.join(words[:3])  # Join the top 3 words with a separator
    print(f'Cluster {cluster}: {heading}')


# %%
from bertopic import BERTopic
from hdbscan import HDBSCAN
# Assume 'df' is your DataFrame and 'processed_text' is the column with text data
# Assume 'cluster' is the column with cluster labels

# Loop through each cluster and apply BERTopic
topic_models = {}
for cluster_id in df['cluster'].unique():
    cluster_texts = df[df['cluster'] == cluster_id]['processed_text'].tolist()
    
    # Create a BERTopic model
    topic_model = BERTopic(hdbscan_model=HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True),
                           language="multilingual", calculate_probabilities=True, verbose=True)

    topics, _ = topic_model.fit_transform(cluster_texts)
    
    # Store the model for later use
    topic_models[cluster_id] = topic_model

    # Optionally, visualize the topic results
    # topic_model.visualize_topics()

# To access a specific model:
# topic_models[some_cluster_id].get_topic_info()

# %%
topic_models[0].get_topic_info()

# %%
