# MovieLens Neural Network Recommender System

This repository demonstrates an end-to-end neural collaborative filtering pipeline, from preprocessing to ranking metrics and embedding-based insight extraction built with TensorFlow to:

- predict user-movie ratings
- generate Top-N personalized movie recommendations
- evaluate both regression quality (RMSE/MAE) and ranking quality (Precision@K, Recall@K, HitRate@K, NDCG@K).

The project includes two notebook workflows:
- `movielens-small-nn-recsys.ipynb` for the MovieLens small dataset (100k+ ratings).
- `movielens-large-nn-recsys.ipynb` for the MovieLens latest large dataset (33M+ ratings).

Both datasets are from GroupLens MovieLens and include their own license and citation details in:
- `data-small/README.txt`
- `data-large/README.txt`
The actual csv files are not included in the repo.

## Project Structure

This is the directory structure, even if not all the files have been uploaded here.
```
movie-lens-recsys/
|- movielens-small-nn-recsys.ipynb
|- movielens-large-nn-recsys.ipynb
|- artifacts/
|  |- large_metrics.json
|  |- large_rec_model.keras
|  |- large_user_embeddings.npy
|  \- large_movie_embeddings.npy
|- data-small/ 
|  |- ratings.csv 
|  |- movies.csv
|  \- README.txt
\- data-large/
   |- ratings.csv
   |- movies.csv
   \- README.txt
```

## Future Additions

- Hyperparameter search (embedding size, depth, learning rate etc).
- Regularization tuning (dropout, stronger weight decay).
- Temporal train/validation splitting (timestamp has been completely avoided as of now).
- Hybrid models combining collaborative and content features.

## Author Notes

