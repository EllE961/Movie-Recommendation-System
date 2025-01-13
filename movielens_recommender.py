import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

class MovieLensRecommender:
    def __init__(self, rating_path, movie_path, model_path="svd_model.pkl"):
        self.rating_path = rating_path
        self.movie_path = movie_path
        self.model_path = model_path
        self.algo = None
        self.trainset = None
        self.movie_map = {}

    def load_data(self):
        ratings_df = pd.read_csv(self.rating_path)
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
        movies_df = pd.read_csv(self.movie_path)
        for row in movies_df.itertuples():
            self.movie_map[row.movieId] = row.title
        return data

    def train_or_load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.algo, self.trainset = pickle.load(f)
            print("Loaded existing model from file.")
        else:
            data = self.load_data()
            trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
            algo = SVD(n_factors=100, random_state=42)
            algo.fit(trainset)
            preds = algo.test(testset)
            rmse_score = accuracy.rmse(preds)
            print("Test RMSE:", rmse_score)
            full_trainset = data.build_full_trainset()
            algo.fit(full_trainset)
            self.algo = algo
            self.trainset = full_trainset
            with open(self.model_path, "wb") as f:
                pickle.dump((self.algo, self.trainset), f)
            print("Trained new model and saved to file.")

    def get_top_n(self, user_id, n=5):
        try:
            inner_uid = self.trainset.to_inner_uid(str(user_id))
        except ValueError:
            raise ValueError(f"User {user_id} is not in the trainset.")
        user_items = set(j for (j, _) in self.trainset.ur[inner_uid])
        all_items = self.trainset.all_items()
        preds = []
        for item in all_items:
            if item not in user_items:
                raw_iid = self.trainset.to_raw_iid(item)
                est_rating = self.algo.predict(str(user_id), str(raw_iid)).est
                preds.append((raw_iid, est_rating))
        preds.sort(key=lambda x: x[1], reverse=True)
        top_n = preds[:n]
        mapped = [(self.movie_map.get(int(mid), f"Movie {mid}"), rating) for (mid, rating) in top_n]
        return mapped

def main():
    rating_path = "ml-32m/ratings.csv"
    movie_path = "ml-32m/movies.csv"
    recommender = MovieLensRecommender(rating_path, movie_path, model_path="svd_model.pkl")
    recommender.train_or_load_model()
    user_id = 1
    top_n = 5
    try:
        recs = recommender.get_top_n(user_id, n=top_n)
        print(f"Top {top_n} recommendations for user {user_id}:")
        for title, rating in recs:
            print(f"{title} (estimated rating: {rating:.2f})")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
