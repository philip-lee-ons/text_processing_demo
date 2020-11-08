# -*- coding: utf-8 -*-
__version__ = '0.1.0'

import configparser
import json
import logging
import os
import string

import nltk
import pandas as pd

CONFIG_FILEPATH = os.path.join(
    os.path.expanduser("~"),
    "text_processing_demo_config.ini"
)


def read_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILEPATH)
    return config


def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def fetch_stopwords():
    nltk.download("stopwords")
    stopwords = nltk.corpus.stopwords.words("english")
    stopword_set = set(word.replace("'", "") for word in stopwords)
    return stopword_set


config = read_config()
logger = setup_logging()

STOPWORDS = fetch_stopwords()


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

    translation_table = str.maketrans('', '', string.punctuation)
    no_punctuation = first_sentence.translate(translation_table)

    lowercase = no_punctuation.lower()

    no_stopwords = " ".join(word for word in lowercase.split(" ") if word not in STOPWORDS)

    return no_stopwords


def clean_overview_col(df: pd.DataFrame, col: str):
    """Prepare overview column for text processing.

    Returns:
        Same dataframe with an additional column for cleaned text.
    """
    logger.info(f"Cleaning column: {col} ")
    df[f"{col}_cleaned"] = df[col].fillna("").apply(clean_overview_text)
    return df


if __name__ == "__main__":

    movies_df = load_movie_data()

    movies_df = clean_overview_col(movies_df, "overview")

