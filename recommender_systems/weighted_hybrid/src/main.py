from load_data import load_metadata, data_preprocessing


def read_csv():
    df_movie = load_metadata("../data/tmdb_5000_movies.csv")
    df_credits = load_metadata("../data/tmdb_5000_credits.csv")
    return df_movie, df_credits


def calculate_weighted_average(df):
    # calculate weighted average of movies based
    v = df["vote_count"]
    r = df["vote_average"]
    c = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.70)

    df["weighted_average"] = ((r * v) + (c * m)) / (v + m)
    return df


if __name__ == "__main__":
    df_movie, df_credits = read_csv()
    print(df_movie.head())
    print(df_credits.head())
    print(df_movie.shape)
    print(df_credits.shape)
    # clean and preprocesses dara
    movies_cleaned_df = data_preprocessing(df_movie, df_credits)
    movies_cleaned_df = calculate_weighted_average(movies_cleaned_df)
    print(movies_cleaned_df.head())
    
    

