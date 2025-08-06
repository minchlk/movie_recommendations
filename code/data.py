import os
import polars as pl
import zipfile, requests, io


if os.path.exists("ml-latest-small"):
        print("Loading dataset...")
else: 
    print("Downloading dataset...")
    response = requests.get("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("ml-latest-small")
    print("Loading dataset...")


ratings_df = pl.read_csv("ml-latest-small/ml-latest-small/ratings.csv")
movies_df = pl.read_csv("ml-latest-small/ml-latest-small/movies.csv")
tags_df = pl.read_csv("ml-latest-small/ml-latest-small/tags.csv")

#clean data and prepare for vectorization
print("Preparing Data")
tags_df2 = tags_df.with_columns(pl.col("tag").str.to_lowercase().alias("tag"))
tags_agg = tags_df2.group_by("movieId").agg(
    pl.col("tag").unique().alias("tags_list")).with_columns(
    pl.col("tags_list").list.join(" ").alias("tags_str")).select(["movieId", "tags_str"])
#print(tags_agg)

movies_df2 = movies_df.with_columns(pl.col("genres").str.to_lowercase().alias("genres"))
movies_df3 = movies_df2.with_columns(pl.col("genres").str.replace_all(r"\|", " ").alias("genres"))
#print(movies_df3)

movie_text_df = movies_df3.join(tags_agg, on="movieId", how="left").with_columns([
    pl.col("genres").str.replace_all(r"\|", " ").alias("genres_str"),
    pl.col("tags_str").fill_null("").alias("tags_str")]).with_columns([
    (pl.col("genres_str") + " " + pl.col("tags_str")).alias("text")]).select([
    "movieId", "title", "text"])
#print(movie_text_df)


movie_ids = movie_text_df["movieId"].to_list()
titles = movie_text_df["title"].to_list()
corpus = movie_text_df["text"].to_list()

