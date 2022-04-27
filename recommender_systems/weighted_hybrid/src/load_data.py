# dataset link: https://www.kaggle.com/tmdb/tmdb-movie-metadata
import pandas as pd
import numpy as np


def load_metadata(path):
    """
    Loads the metadata from the specified path.
    """
    metadata = pd.read_csv(path)
    return metadata


def data_preprocessing(df_movie, df_credits):
    """
    Preprocesses the data.
    """
    credit_column_renamed = df_credits.rename(index=str, columns={"movie_id": "id"})
    movies_df_merged = df_movie.merge(credit_column_renamed, on="id")
    movies_cleaned_df = movies_df_merged.drop(
        columns=["homepage", "title_x", "title_y", "status", "production_countries"]
    )
    return movies_cleaned_df






