!wget https://lazyprogrammer.me/course_files/nlp/tmdb_5000_movies.csv

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

df = pd.read_csv(
    "tmdb_5000_movies.csv",
    usecols=["title", "overview", "genres", "keywords", "tagline"],
)
df = df.dropna()
df[["title", "overview", "tagline", "genres", "keywords"]]


def genres_and_keywords_to_string(row):
    genres = " ".join(
        [genre["name"].replace(" ", "") for genre in json.loads(row["genres"])]
    )
    keywords = " ".join(
        [keyword["name"].replace(" ", "") for keyword in json.loads(row["keywords"])]
    )
    tagline = row["tagline"] if pd.notnull(row["tagline"]) else ""
    return f"{genres} {keywords} {tagline}"


df["genres_keywords_tagline"] = df.apply(genres_and_keywords_to_string, axis=1)
df[["title", "overview", "genres_keywords_tagline"]]

df["combined_text"] = df["overview"] + " " + df["genres_keywords_tagline"]
df[["title", "combined_text"]]

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    clean_text = text.lower()
    words = [
        lemmatizer.lemmatize(word)
        for word in word_tokenize(clean_text)
        if word.isalnum() and word not in stop_words
    ]
    return " ".join(words)


df["clean_combined_text"] = df["combined_text"].apply(clean_text)
df[["title", "combined_text", "clean_combined_text"]]

pretrained_w2v = api.load("word2vec-google-news-300")
pretrained_glove = api.load("glove-wiki-gigaword-300")
pretrained_fasttext = api.load("fasttext-wiki-news-subwords-300")


def calculate_sentence_vector(sentence, model):
    words = [word for word in sentence.split() if word in model]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean([model[word] for word in words], axis=0)


df["word2vec_vector"] = df["clean_combined_text"].apply(
    lambda x: calculate_sentence_vector(x, pretrained_w2v)
)
df["glove_vector"] = df["clean_combined_text"].apply(
    lambda x: calculate_sentence_vector(x, pretrained_glove)
)
df["fasttext_vector"] = df["clean_combined_text"].apply(
    lambda x: calculate_sentence_vector(x, pretrained_fasttext)
)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["clean_combined_text"])
df["tfidf_vector"] = list(tfidf_matrix.toarray())


def plot_similarity_scores(title, vector_column, model_name):
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    idx = indices.get(title)

    if idx is None:
        print(f"Movie titled '{title}' not found.")
        return

    query_vec = df.loc[idx, vector_column].reshape(1, -1)
    cosine_similarities = cosine_similarity(
        query_vec, np.vstack(df[vector_column].values)
    )

    sim_scores = list(enumerate(cosine_similarities[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in sim_scores[:6]]
    plt.barh(
        df.iloc[movie_indices]["title"],
        [sim_scores[i][1] for i in range(6)],
        color="skyblue",
    )
    plt.xlabel("Cosine Similarity")
    plt.title(f"Top Similar Movies based on {model_name}")
    plt.show()


def get_recommendations(title, vector_column):
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    idx = indices.get(title)

    if idx is None:
        return f"Movie titled '{title}' not found."

    query_vec = df.loc[idx, vector_column].reshape(1, -1)
    cosine_similarities = cosine_similarity(
        query_vec, np.vstack(df[vector_column].values)
    )

    sim_scores = list(enumerate(cosine_similarities[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = df.iloc[movie_indices]["title"]

    return recommended_movies, sim_scores


def compare_recommendations(title):
    print(f"For the movie: {title}")

    models = {
        "Word2Vec": "word2vec_vector",
        "GloVe": "glove_vector",
        "FastText": "fasttext_vector",
        "TF-IDF": "tfidf_vector",
    }

    for model_name, vector_column in models.items():
        recommended_movies, sim_scores = get_recommendations(title, vector_column)
        if isinstance(recommended_movies, str):
            print(recommended_movies)
            continue
        print(f"\nMovies recommended using {model_name}:")
        for i, (movie, score) in enumerate(zip(recommended_movies, sim_scores), 1):
            print(f"{i}. {movie} - Score: {score[1]:.4f}")


query_film = "The Matrix"

plot_similarity_scores(query_film, "word2vec_vector", "Word2Vec")
plot_similarity_scores(query_film, "glove_vector", "GloVe")
plot_similarity_scores(query_film, "fasttext_vector", "FastText")
plot_similarity_scores(query_film, "tfidf_vector", "TF-IDF")

compare_recommendations(query_film)
