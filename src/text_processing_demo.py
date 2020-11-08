# -*- coding: utf-8 -*-
"""Text processing demo

Simplified version of a text processing pipeline.

    * load data
    * clean text
    * vectorize (with a word embedding)
    * reduce dimension
    * cluster

Sources:
    https://www.kaggle.com/tmdb/tmdb-movie-metadata
    https://fasttext.cc/docs/en/pretrained-vectors.html
    https://radimrehurek.com/gensim_3.8.3/models/fasttext.html
    https://radimrehurek.com/gensim_3.8.3/auto_examples/tutorials/run_fasttext.html

Pre-trained FastText:
    https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
"""
__version__ = "0.1.0"

import configparser
import json
import logging
import functools
import os
import string

import gensim
import hdbscan
import nltk
import numpy as np
import pandas as pd
import sklearn

CONFIG_FILEPATH = os.path.join(
    os.path.expanduser("~"), "text_processing_demo_config.ini"
)


def _read_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILEPATH)
    return config


def _setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def fetch_stopwords():
    nltk.download("stopwords")
    stopwords = nltk.corpus.stopwords.words("english")
    stopword_set = set(word.replace("'", "") for word in stopwords)
    return stopword_set


config = _read_config()
logger = _setup_logging()

STOPWORDS = fetch_stopwords()
RANDOM_STATE = 3052528580


def load_movie_data():
    """Loads Movie data.

    Location set in config.ini.

    Returns:
        pd.DataFrame: Movies
            title: Title
            overview: Short description
            genres: Sorted tuple of genres
    """

    def convert_genres(genres):
        """Json dict list to tuple of names
        Genre comes in as json:

            [
                {"id": 1, "name": "Some genre"},
                {"id": 2, "name": "Another genre"
            ]

        Convert to: ("Another genre", "Some genre")
        """
        genres_dicts = json.loads(genres)
        genre_names = [d["name"] for d in genres_dicts]
        genres_out = tuple(sorted(genre_names))
        return genres_out

    filepath = config["filepaths"]["MovieData"]
    logger.info(f"Reading Movie data from {filepath}")

    df = pd.read_csv(filepath, usecols=["original_title", "overview", "genres"])

    logger.debug("Converting genre format")
    df.genres = df.genres.apply(convert_genres)

    logger.debug("Fixing column name")
    df = df.rename(columns={"original_title": "title"})

    return df[["title", "overview", "genres"]]


def clean_overview_text(text: str):
    """Prepare overview text for further processing.

    * keep first sentence only
    * remove punctuation
    * lowercase
    * no stopwords
    """
    text = text.replace("!", ".").replace("?", ".")
    first_sentence = text.split(".")[0]

    translation_table = str.maketrans("", "", string.punctuation)
    no_punctuation = first_sentence.translate(translation_table)

    lowercase = no_punctuation.lower()

    no_stopwords = " ".join(
        word for word in lowercase.split(" ") if word not in STOPWORDS
    )

    return no_stopwords


def clean_overview_col(df: pd.DataFrame, col: str):
    """Prepare overview column for text processing.

    Returns:
        Same dataframe with an additional column for cleaned text.
    """
    logger.info(f"Cleaning column: {col} ")
    df[f"{col}_cleaned"] = df[col].fillna("").apply(clean_overview_text)
    return df


def fetch_fasstext_pretrained():
    filepath = config["filepaths"]["FastTextPretrainedBinary"]
    logger.info(f"Loading FastText pretrained from {filepath}")
    wv = gensim.models.fasttext.load_facebook_vectors(filepath)

    logger.info("Model loaded")
    return wv


def vectorize_text(
    wv: gensim.models.keyedvectors.WordEmbeddingsKeyedVectors, text: str
):
    """Apply word vectorizer to text.

    This takes a simple averaging approach
    i.e. every word in the text is passed to the model and the resulting
    vectors are averaged.
    """
    vecs = np.array([wv[word] for word in text.split(" ")])

    return np.mean(vecs, axis=0)


def reduce_dimensionality(vector_col: pd.Series):
    # There have been issues with the umap import
    import umap

    logger.info("Applying umap to reduce dimension")
    vecs = np.array(list(vector_col.values))

    clusterable_embedding = umap.UMAP(
        n_neighbors=5,
        min_dist=0.0,
        n_components=10,
        random_state=RANDOM_STATE,
        verbose=10,
    ).fit_transform(vecs)

    return pd.Series(data=clusterable_embedding.tolist(), index=vector_col.index)


def cluster(vector_col: pd.Series):

    vecs = np.array(list(vector_col))

    labels = hdbscan.HDBSCAN().fit_predict(vecs)

    return pd.Series(data=labels.tolist(), index=vector_col.index)


def evaluate(df, original, labels):
    """Cluster similarity"""
    labels_true = df[original]
    labels_pred = df[labels]
    result = sklearn.metrics.cluster.adjusted_rand_score(
        labels_true=labels_true, labels_pred=labels_pred
    )

    logger.info("Are these labellings anything like each other?")
    logger.info(f"Adjusted rand score: {result}")


def plot_embedding(df):
    import matplotlib.pyplot as plt
    import seaborn
    import sklearn

    target = sklearn.preprocessing.LabelEncoder().fit_transform(df.genres)

    vecs = np.array(list(df["overview_cleaned_vectorized_low_dimension"].values))

    plt.scatter(vecs[:, 0], vecs[:, 1], c=target, s=0.1, cmap="Spectral")

    plt.show()


if __name__ == "__main__":

    movies_df = load_movie_data()

    movies_df = clean_overview_col(movies_df, "overview")

    wv = fetch_fasstext_pretrained()

    text_to_vec = functools.partial(vectorize_text, wv)

    movies_df["overview_cleaned_vectorized"] = movies_df.overview_cleaned.apply(
        text_to_vec
    )

    movies_df["overview_cleaned_vectorized_low_dimension"] = reduce_dimensionality(
        movies_df.overview_cleaned_vectorized
    )

    movies_df["label"] = cluster(movies_df.overview_cleaned_vectorized_low_dimension)

    plot_embedding(movies_df)
