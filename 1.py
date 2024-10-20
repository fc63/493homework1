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
df["combined_text"] = df["overview"] + " " + df["genres_keywords_tagline"]

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

pretrained_w2v = api.load("word2vec-google-news-300")
pretrained_glove = api.load("glove-wiki-gigaword-300")
pretrained_fasttext = api.load("fasttext-wiki-news-subwords-300")

"""
burada mesela örnek olarak bir sorgunun vektörel değeri nasıl hesaplanır diye baktığımızda
öncelikle pretrained_w2v[word] ile Word2Vec modelinden kelimenin 300 boyutlu vektörü çekiliyor
(bu vektör eğitilen modele göre değişiklik gösteriyor)
çekilen vektör; modelin eğitildiği veri kümesi üzerinde, kelimenin diğer kelimelerle olan ilişkisini yansıtan sayısal bir temsildir
Kelimenin bağlamı, yakınındaki diğer kelimelerden etkilenerek bu vektörle ifade ediliyor.
yazdığımız get_vector fonksiyonu ise her kelimenin vektörünü modelden alıyor ve bu vektörlerin ortalamasını alarak 
verdiğimiz text'in genel vektörünü oluşturuyor.
o da şöyle ki; kelimeler tek tek vektör olarak modelden alınıp listeye ekleniyor,
ve daha sonra bu liste içerisindeki tüm kelime vektörlerinin ortalaması alınarak nihai vektör (yani word2vec_vector) hesaplanıyor.
kelime vektörlerinin modelde birbirine olan uzaklığı, Word2Vec'in eğitildiği süreçte öğrenilen anlamsal ilişkilerle belirlenmiştir.
yani, modelden alınan her vektör aslında o kelimenin diğer kelimelere olan "anlamsal uzaklıklarını" içeren bir temsildir.
"""


def get_vector(text, model):
    words = text.split()
    word_vecs = [model[word] for word in words if word in model]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(300)


df["word2vec_vector"] = df["clean_combined_text"].apply(
    lambda text: get_vector(text, pretrained_w2v)
)
df["glove_vector"] = df["clean_combined_text"].apply(
    lambda text: get_vector(text, pretrained_glove)
)
df["fasttext_vector"] = df["clean_combined_text"].apply(
    lambda text: get_vector(text, pretrained_fasttext)
)

tfidf_vectorizer = TfidfVectorizer(max_features=2000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_combined_text"])
df["tfidf_vector"] = list(tfidf_matrix.toarray())


query_film = "The Matrix"


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
    scores = cosine_similarities.flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(scores, label=model_name)
    plt.title(f"Cosine Similarity Scores for {title} using {model_name}")
    plt.xlabel("Movies")
    plt.ylabel("Cosine Similarity Score")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_similarity_scores(query_film, "word2vec_vector", "Word2Vec")
plot_similarity_scores(query_film, "glove_vector", "GloVe")
plot_similarity_scores(query_film, "fasttext_vector", "FastText")
plot_similarity_scores(query_film, "tfidf_vector", "TF-IDF")


def plot_sorted_similarity_scores(title, vector_column, model_name):
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    idx = indices.get(title)

    if idx is None:
        print(f"Movie titled '{title}' not found.")
        return

    query_vec = df.loc[idx, vector_column].reshape(1, -1)
    cosine_similarities = cosine_similarity(
        query_vec, np.vstack(df[vector_column].values)
    )
    scores = cosine_similarities.flatten()
    sorted_scores = scores[(-scores).argsort()]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_scores, label=f"Sorted {model_name} Similarity Scores")
    plt.title(f"Cosine Similarity Scores for {title} (Sorted) using {model_name}")
    plt.xlabel("Movies (Sorted by Similarity)")
    plt.ylabel("Cosine Similarity Score")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_sorted_similarity_scores(query_film, "word2vec_vector", "Word2Vec")
plot_sorted_similarity_scores(query_film, "glove_vector", "GloVe")
plot_sorted_similarity_scores(query_film, "fasttext_vector", "FastText")
plot_sorted_similarity_scores(query_film, "tfidf_vector", "TF-IDF")


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


compare_recommendations(query_film)
