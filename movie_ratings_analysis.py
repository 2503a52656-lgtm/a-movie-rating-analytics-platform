"""
Complete Movie Ratings Data Analysis
------------------------------------
Input : cleaned_ratings(2).csv
Output: analysis CSV files, charts, and an Excel workbook.

Install required packages:
    pip install pandas numpy matplotlib seaborn openpyxl

Run:
    python movie_ratings_analysis.py

Place this script in the same folder as cleaned_ratings(2).csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- 1. SETTINGS --------------------

INPUT_FILE = Path("cleaned_ratings(2).csv")
OUTPUT_DIR = Path("movie_ratings_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- 2. LOAD DATA --------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}. Put the CSV file in the same folder "
        "as this Python script, or update INPUT_FILE."
    )

df = pd.read_csv(INPUT_FILE)

print("\n========== DATASET OVERVIEW ==========")
print("First five rows:\n", df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nDataset information:")
df.info()

# -------------------- 3. VALIDATE COLUMNS --------------------

required_columns = {"user_id", "movie_id", "rating"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")

# Convert numeric columns safely
for column in ["user_id", "movie_id", "rating"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Use rated_at when available; otherwise derive it from Unix timestamp.
if "rated_at" in df.columns:
    df["rated_at"] = pd.to_datetime(df["rated_at"], errors="coerce", utc=True)
elif "timestamp" in df.columns:
    df["rated_at"] = pd.to_datetime(
        df["timestamp"], unit="s", errors="coerce", utc=True
    )
else:
    df["rated_at"] = pd.NaT

# -------------------- 4. DATA QUALITY --------------------

quality = pd.DataFrame({
    "check": [
        "Total rows",
        "Total columns",
        "Missing cells",
        "Duplicate complete rows",
        "Duplicate user-movie pairs",
        "Unique users",
        "Unique movies",
        "Invalid/missing user IDs",
        "Invalid/missing movie IDs",
        "Invalid/missing ratings",
    ],
    "result": [
        len(df),
        len(df.columns),
        int(df.isna().sum().sum()),
        int(df.duplicated().sum()),
        int(df.duplicated(subset=["user_id", "movie_id"]).sum()),
        int(df["user_id"].nunique()),
        int(df["movie_id"].nunique()),
        int(df["user_id"].isna().sum()),
        int(df["movie_id"].isna().sum()),
        int(df["rating"].isna().sum()),
    ],
})
quality.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

print("\n========== DATA QUALITY ==========")
print(quality.to_string(index=False))

# Keep valid rating rows for rating-based analysis.
analysis_df = df.dropna(subset=["user_id", "movie_id", "rating"]).copy()

# -------------------- 5. DESCRIPTIVE STATISTICS --------------------

stats = analysis_df["rating"].describe().to_frame(name="rating")
stats.to_csv(OUTPUT_DIR / "rating_statistics.csv")

print("\n========== RATING STATISTICS ==========")
print(stats)

rating_mean = analysis_df["rating"].mean()
rating_median = analysis_df["rating"].median()
rating_mode = analysis_df["rating"].mode()
rating_mode = rating_mode.iloc[0] if not rating_mode.empty else np.nan

# -------------------- 6. RATING DISTRIBUTION --------------------

rating_distribution = (
    analysis_df.groupby("rating")
    .size()
    .rename("rating_count")
    .reset_index()
    .sort_values("rating")
)
rating_distribution["percentage"] = (
    rating_distribution["rating_count"] / len(analysis_df) * 100
)
rating_distribution.to_csv(OUTPUT_DIR / "rating_distribution.csv", index=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=rating_distribution, x="rating", y="rating_count")
plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rating_distribution.png", dpi=300)
plt.show()

# -------------------- 7. MOVIE ANALYSIS --------------------

movie_stats = (
    analysis_df.groupby("movie_id")["rating"]
    .agg(
        rating_count="count",
        average_rating="mean",
        median_rating="median",
        rating_std="std",
    )
    .reset_index()
)

top_movies = movie_stats.sort_values(
    ["rating_count", "average_rating"], ascending=[False, False]
)
top_movies.head(20).to_csv(OUTPUT_DIR / "top_20_most_rated_movies.csv", index=False)

# Use a minimum rating-count threshold to reduce unreliable averages.
highest_rated = (
    movie_stats[movie_stats["rating_count"] >= 100]
    .sort_values(["average_rating", "rating_count"], ascending=[False, False])
)
highest_rated.head(20).to_csv(
    OUTPUT_DIR / "top_20_highest_rated_movies_min_100.csv", index=False
)

print("\n========== TOP 10 MOST-RATED MOVIES ==========")
print(top_movies.head(10).to_string(index=False))

print("\n========== TOP 10 HIGHEST-RATED MOVIES (AT LEAST 100 RATINGS) ==========")
print(highest_rated.head(10).to_string(index=False))

# Plot top 10 most-rated movies
plot_movies = top_movies.head(10).sort_values("rating_count")
plt.figure(figsize=(9, 6))
plt.barh(plot_movies["movie_id"].astype(str), plot_movies["rating_count"])
plt.title("Top 10 Most-Rated Movies (Movie IDs)")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie ID")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_10_most_rated_movies.png", dpi=300)
plt.show()

# -------------------- 8. USER ACTIVITY --------------------

user_stats = (
    analysis_df.groupby("user_id")["rating"]
    .agg(rating_count="count", average_rating="mean")
    .reset_index()
    .sort_values(["rating_count", "average_rating"], ascending=[False, False])
)
user_stats.to_csv(OUTPUT_DIR / "user_activity.csv", index=False)

print("\n========== TOP 10 ACTIVE USERS ==========")
print(user_stats.head(10).to_string(index=False))

# -------------------- 9. DATE / YEAR / MONTH ANALYSIS --------------------

dated_df = analysis_df.dropna(subset=["rated_at"]).copy()
dated_df["year"] = dated_df["rated_at"].dt.year
dated_df["month"] = dated_df["rated_at"].dt.month
dated_df["year_month"] = dated_df["rated_at"].dt.strftime("%Y-%m")
dated_df["weekday"] = dated_df["rated_at"].dt.day_name()

if not dated_df.empty:
    yearly_trend = (
        dated_df.groupby("year").size().rename("rating_count").reset_index()
    )
    yearly_trend.to_csv(OUTPUT_DIR / "yearly_rating_trend.csv", index=False)

    monthly_trend = (
        dated_df.groupby(["year", "month", "year_month"])
        .size()
        .rename("rating_count")
        .reset_index()
    )
    monthly_trend.to_csv(OUTPUT_DIR / "monthly_rating_trend.csv", index=False)

    weekday_trend = (
        dated_df.groupby("weekday").size().rename("rating_count").reset_index()
    )
    weekday_trend.to_csv(OUTPUT_DIR / "weekday_rating_trend.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(yearly_trend["year"], yearly_trend["rating_count"], marker="o")
    plt.title("Number of Ratings by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Ratings")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "yearly_rating_trend.png", dpi=300)
    plt.show()

    plt.figure(figsize=(11, 5))
    plt.plot(monthly_trend["year_month"], monthly_trend["rating_count"])
    plt.title("Number of Ratings by Month")
    plt.xlabel("Year-Month")
    plt.ylabel("Number of Ratings")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_rating_trend.png", dpi=300)
    plt.show()
else:
    yearly_trend = pd.DataFrame(columns=["year", "rating_count"])
    monthly_trend = pd.DataFrame(columns=["year", "month", "year_month", "rating_count"])
    weekday_trend = pd.DataFrame(columns=["weekday", "rating_count"])
    print("\nNo valid date column found; date trend analysis was skipped.")

# -------------------- 10. OVERALL SUMMARY --------------------

summary_rows = [
    ("Total valid ratings", len(analysis_df)),
    ("Unique users", analysis_df["user_id"].nunique()),
    ("Unique movies", analysis_df["movie_id"].nunique()),
    ("Average rating", rating_mean),
    ("Median rating", rating_median),
    ("Most common rating", rating_mode),
    ("Minimum rating", analysis_df["rating"].min()),
    ("Maximum rating", analysis_df["rating"].max()),
    ("Rating standard deviation", analysis_df["rating"].std()),
    ("Ratings >= 3.5 (%)", (analysis_df["rating"] >= 3.5).mean() * 100),
    ("Ratings with valid dates", len(dated_df)),
]

if not dated_df.empty:
    summary_rows.extend([
        ("First rating date (UTC)", dated_df["rated_at"].min().strftime("%Y-%m-%d %H:%M:%S")),
        ("Last rating date (UTC)", dated_df["rated_at"].max().strftime("%Y-%m-%d %H:%M:%S")),
        ("Year with most ratings", int(yearly_trend.loc[yearly_trend["rating_count"].idxmax(), "year"])),
        ("Highest yearly rating count", int(yearly_trend["rating_count"].max())),
    ])

summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
summary.to_csv(OUTPUT_DIR / "analysis_summary.csv", index=False)

print("\n========== ANALYSIS SUMMARY ==========")
print(summary.to_string(index=False))

# -------------------- 11. EXPORT EXCEL WORKBOOK --------------------

excel_path = OUTPUT_DIR / "Complete_Movie_Ratings_Analysis.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    quality.to_excel(writer, sheet_name="Data Quality", index=False)
    analysis_df.to_excel(writer, sheet_name="Cleaned Data", index=False)
    rating_distribution.to_excel(writer, sheet_name="Rating Distribution", index=False)
    top_movies.head(20).to_excel(writer, sheet_name="Most Rated Movies", index=False)
    highest_rated.head(20).to_excel(writer, sheet_name="Highest Rated Movies", index=False)
    user_stats.to_excel(writer, sheet_name="User Activity", index=False)
    yearly_trend.to_excel(writer, sheet_name="Yearly Trend", index=False)
    monthly_trend.to_excel(writer, sheet_name="Monthly Trend", index=False)
    weekday_trend.to_excel(writer, sheet_name="Weekday Trend", index=False)

    # Basic formatting and filters for each sheet
    for sheet in writer.book.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = cell.font.copy(bold=True, color="FFFFFF")
            cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
        for column_cells in sheet.columns:
            max_length = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in column_cells[:500]
            )
            sheet.column_dimensions[column_cells[0].column_letter].width = min(
                max(max_length + 2, 12), 35
            )

print("\n==============================================")
print("DATA ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"All output files saved in: {OUTPUT_DIR.resolve()}")
print(f"Excel report: {excel_path.resolve()}")
print("==============================================")
