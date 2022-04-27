import pandas as pd
import argparse
from tqdm import tqdm
import os
import pickle
import time

tqdm.pandas()

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from eval import evaluation_main


def preprocess_data(df):
    # df["text"] = df["title"] + " " + df["description"]
    df["kws"] = df["kws"].astype(str)
    # select only one or two categories
    # category_list = ["Tea", "Coffee"]
    # df = df[df["category_4"].isin(category_list)]
    # df.reset_index(drop=True, inplace=True)
    # print("Removing stop words")
    # df["text"] = df["text"].progress_apply(
    #     lambda x: " ".join(x for x in x.split() if x not in stopwords.words("english"))
    # )
    # lemmatizer = WordNetLemmatizer()
    # print("Lemmatizing")
    # df["text"] = df["text"].progress_apply(
    #     lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
    # )
    # # lower all the words
    # print("Lowering all the words")
    # df["text"] = df["text"].progress_apply(
    #     lambda x: " ".join(x.lower() for x in x.split())
    # )

    # # remove all numbers from the text field
    # print("Removing all numbers")
    # df["text"] = df["text"].progress_apply(
    #     lambda x: " ".join(x for x in x.split() if not x.isdigit())
    # )

    return df


def vectorize_data(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["kws"])
    return X, vectorizer


def define_model(k, X):
    true_k = k
    # Running model with 15 different centroid initializations & maximum iterations are 500
    model = KMeans(n_clusters=true_k, init="k-means++", max_iter=500, n_init=2)
    model.fit(X)
    return model


def top_terms_per_cluster(model, vectorizer, n_terms=15):
    # save top 15 terms per cluster in a csv file
    terms = vectorizer.get_feature_names()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms_per_cluster = []
    for i in range(model.n_clusters):
        terms_per_cluster.append([terms[ind] for ind in order_centroids[i, :n_terms]])
    return terms_per_cluster


def cluster_predict(str_input, vectorizer, model):
    Y = vectorizer.transform(list(str_input))
    prediction = model.predict(Y)
    return prediction


def recommend_util(str_input, df, vectorizer, model):

    # match on the basis course-id and form whole 'Description' entry out of it.
    temp_df = df.loc[df["asin"] == str_input]
    temp_df["InputString"] = temp_df["kws"]
    str_input = list(temp_df["InputString"])

    # Predict category of input string category
    prediction_inp = cluster_predict(str_input, vectorizer, model)
    prediction_inp = int(prediction_inp)
    # Based on the above prediction 10 random courses are recommended from the whole data-frame
    # Recommendation Logic is kept super-simple for current implementation.
    temp_df = df.loc[df["ClusterPrediction"] == prediction_inp]
    temp_df = temp_df.head(10)

    return list(temp_df["asin"])


def make_recommendation(asin, df, vectorizer, model):
    return recommend_util(asin, df, vectorizer, model)


def get_recommend_product_asin_category(df, list):
    category_list = []
    for i in list or []:
        category_list.append(
            df[df["asin"] == i]["category_4"].to_string(index=False)
        )  # TODO: item()/
    return category_list


def main():
    data = pd.read_excel("recommendation_output.xlsx")
    df = pd.read_csv("flavor_taste_ingredients.csv")
    # df = df[
    #     [
    #         "asin",
    #         "title",
    #         "description",
    #         "category",
    #         "available_also_buy",
    #         "category_4",
    #         "available_also_buy_category",
    #     ]
    # ]
    df = preprocess_data(df)
    X, vectorizer = vectorize_data(df)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--true_k",
        "-k",
        type=int,
        required=True,
        help="number of clusters",
    )

    args = parser.parse_args()
    cluster_number = args.true_k

    model = define_model(cluster_number, X)

    terms_per_cluster = top_terms_per_cluster(model, vectorizer)
    # save terms_per_cluster in a csv file
    df_terms = pd.DataFrame(terms_per_cluster)

    RESULT_SAVE_PATH = f"result_k{cluster_number}"
    os.makedirs(RESULT_SAVE_PATH, exist_ok=True)

    df_terms.to_csv(f"{RESULT_SAVE_PATH}/top_terms.csv", index=False)

    # save model in a pickle file
    MODEL_SAVE_PATH = f"models/finalized_model_k{cluster_number}.sav"
    pickle.dump(model, open(MODEL_SAVE_PATH, "wb"))

    df["ClusterPrediction"] = ""
    df["InputString"] = df["kws"]
    print("Predicting cluster for each input string")
    df["ClusterPrediction"] = df.apply(
        lambda x: cluster_predict(df["InputString"], vectorizer, model), axis=0
    )

    print("Making recommendation for each input string")
    df["recommend_product_asin"] = df.progress_apply(
        lambda x: make_recommendation(x["asin"], df, vectorizer, model), axis=1
    )

    final_df = df[
        [
            "asin",
            "title",
            "description",
            "category_4",
            "available_also_buy",
            "available_also_buy_category",
            "recommend_product_asin",
            "ClusterPrediction",
        ]
    ]

    print("Finding category of each predicted product")
    final_df["recommend_product_asin_category"] = final_df.progress_apply(
        lambda x: get_recommend_product_asin_category(
            data, x["recommend_product_asin"]
        ),
        axis=1,
    )

    final_df.to_csv("predictions.csv", index=False)

    # model evaluation
    print("Evaluating model")
    evaluation_main(RESULT_SAVE_PATH)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Time taken:")
    print(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))
