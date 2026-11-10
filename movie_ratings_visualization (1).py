# Movie Ratings Dataset - Data Visualization
# Install required libraries:
# pip install pandas matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Keep this Python file and cleaned_ratings(3).csv in the same folder.
file_path = Path("cleaned_ratings(3).csv")

# Load dataset
df = pd.read_csv(file_path)

# Convert date column to datetime
df["rated_at"] = pd.to_datetime(df["rated_at"], errors="coerce", utc=True)

# Folder for saving graph images
output_folder = Path("ratings_visualizations")
output_folder.mkdir(exist_ok=True)


# 1. Distribution of Movie Ratings
rating_counts = df["rating"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(rating_counts.index.astype(str), rating_counts.values)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.tight_layout()
plt.savefig(output_folder / "01_rating_distribution.png", dpi=160)
plt.show()


# 2. Top 15 Most Active Users
top_users = df["user_id"].value_counts().head(15).sort_values()

plt.figure(figsize=(9, 6))
plt.barh(top_users.index.astype(str), top_users.values)
plt.title("Top 15 Most Active Users")
plt.xlabel("Number of Ratings")
plt.ylabel("User ID")
plt.tight_layout()
plt.savefig(output_folder / "02_top_users.png", dpi=160)
plt.show()


# 3. Top 15 Most Rated Movies
top_movies = df["movie_id"].value_counts().head(15).sort_values()

plt.figure(figsize=(9, 6))
plt.barh(top_movies.index.astype(str), top_movies.values)
plt.title("Top 15 Most Rated Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie ID")
plt.tight_layout()
plt.savefig(output_folder / "03_top_movies.png", dpi=160)
plt.show()


# 4. Number of Ratings by Year
valid_dates = df.dropna(subset=["rated_at"]).copy()
valid_dates["year"] = valid_dates["rated_at"].dt.year
year_counts = valid_dates.groupby("year").size()

plt.figure(figsize=(10, 5))
plt.plot(year_counts.index, year_counts.values, marker="o")
plt.title("Number of Ratings by Year")
plt.xlabel("Year")
plt.ylabel("Number of Ratings")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_folder / "04_ratings_by_year.png", dpi=160)
plt.show()


# 5. Average Movie Rating by Year
year_average = valid_dates.groupby("year")["rating"].mean()

plt.figure(figsize=(10, 5))
plt.plot(year_average.index, year_average.values, marker="o")
plt.title("Average Movie Rating by Year")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.ylim(0, 5)
plt.grid(True)
plt.tight_layout()
plt.savefig(output_folder / "05_average_rating_by_year.png", dpi=160)
plt.show()


# Display dataset summary
print("DATASET SUMMARY")
print("-------------------------")
print("Total ratings:", len(df))
print("Unique users:", df["user_id"].nunique())
print("Unique movies:", df["movie_id"].nunique())
print("Average rating:", round(df["rating"].mean(), 2))
print("\nAll five graphs have been saved in:", output_folder.resolve())
