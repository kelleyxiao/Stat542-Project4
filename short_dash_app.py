import pandas as pd
import numpy as np
import requests
from logging import debug
from typing import Dict

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from dash.dependencies import ALL, State


ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', engine = 'python', header=None)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
movies = pd.read_csv('./ml-1m/movies.dat', sep='::', engine = 'python', encoding="ISO-8859-1", header = None)
movies.columns = ['MovieID', 'Title', 'Genres']

multiple_idx = pd.Series([("|" in movie) for movie in movies['Genres']])
movies.loc[multiple_idx, 'Genres'] = 'Multiple'

#Merge ratings and movies DataFrames:
rating_merged = ratings.merge(movies, left_on = 'MovieID', right_on = 'MovieID')
rating_merged.to_csv('rating.csv', index=False)
rating_merged = rating_merged.rename(columns={'MovieID': 'movie_id', 'Title': 'title'})

# System I Recommendation Based on Genres
def top_movies_in_genre(df, genre, min_reviews=10):
    # Filter the DataFrame for the given genre
    genre_df = df[df['Genres'].str.contains(genre)]

    # Group by movie and calculate average rating and number of reviews
    grouped = genre_df.groupby('movie_id')['Rating'].agg(['mean', 'count'])

    # Filter movies with a minimum number of reviews
    qualified = grouped[grouped['count'] >= min_reviews]

    # Sort movies by average rating and get the top 5
    top_movies = qualified.sort_values(by='mean', ascending=False).head(5)

    # Merge with the original DataFrame to get the titles
    top_movies_with_title = top_movies.merge(df[['movie_id', 'title']].drop_duplicates(), on='movie_id')

    return top_movies_with_title

# System II
ratings_df = pd.read_csv('./ratings_df.csv')

pruned_similarity_matrix = np.load('./pruned_similarity_matrix_short.npy')

# Function definition
def myIBCF(newuser, pruned_similarity_matrix, ratings_df):
    predictions = np.zeros(len(newuser))
    # Iterate over each movie to predict the rating
    for l in range(len(newuser)):
        if np.isnan(newuser[l]):  # Skip if the user has already rated the movie
            # Compute prediction for movie l
            neighbors = pruned_similarity_matrix[l, :]
            valid_neighbors = ~np.isnan(neighbors) & ~np.isnan(newuser)
            numerator = np.sum(neighbors[valid_neighbors] * newuser[valid_neighbors])
            denominator = np.sum(neighbors[valid_neighbors])

            if denominator != 0:
                predictions[l] = numerator / denominator

    # Recommend top 10 movies based on predictions
    top_10_indices = np.argsort(-predictions)[:10]
    top_10_movies = [ratings_df.columns[i] for i in top_10_indices if not np.isnan(predictions[i])]
    #top_10_predictions = predictions[top_10_indices]
    recommendations = pd.DataFrame({'movie_id': top_10_movies})
    recommendations['movie_id'] = recommendations['movie_id'].str.replace('m', '')
    recommendations['movie_id'] = pd.to_numeric(recommendations['movie_id'], errors='coerce')
    recommendations = recommendations.merge(movies, left_on = 'movie_id', right_on = 'movie_id')
    recommendations  = recommendations[["movie_id","title","genres"]]
    
    return recommendations

##############################################################################
#####################################Dash#####################################
##############################################################################
# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

def get_displayed_movies():
    return movies.head(100)

def get_recommended_movies(new_user_ratings):
    return movies.head(10)

def get_popular_movies(genre: str):
    if genre == genres[1]:
        return movies.head(10)
    else: 
        return movies[10:20]

#######################################################################################
# from myfuns import (genres, get_displayed_movies, get_popular_movies, get_recommended_movies)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], 
               suppress_callback_exceptions=True)
server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H3("Choose your way to start:", className="display-8"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("By Genre", href="/", active="exact"),
                dbc.NavLink("By Rating", href="/system-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])


def render_page_content(pathname):
    if pathname == "/":
        return html.Div(
            [
                html.H1("Select a genre"),
                dcc.Dropdown(
                    id="genre-dropdown",
                    options=[{"label": k, "value": k} for k in genres],
                    value=None,
                    className="mb-4",
                ),
                html.Div(id="genre-output", className=""),
            ]
        )
    elif pathname == "/system-2":
        movies = get_displayed_movies()
        return html.Div(
            [
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H1("Please rate as many as you can to"),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        children=[
                                            "Get recommendations ",
                                            html.I(className="bi bi-emoji-heart-eyes-fill"),
                                        ],
                                        size="lg",
                                        className="btn-success",
                                        id="button-recommend",
                                    ),
                                    className="p-0",
                                ),
                            ],
                            className="sticky-top bg-white py-2",
                        ),
                        html.Div(
                            [
                                get_movie_card(movie, with_rating=True)
                                for idx, movie in movies.iterrows()
                            ],
                            className="row row-cols-1 row-cols-5",
                            id="rating-movies",
                        ),
                    ],
                    id="rate-movie-container",
                ),
                html.H1(
                    "Your recommendations", id="your-recommendation",  style={"display": "none"}
                ),
                dcc.Loading(
                    [
                        dcc.Link(
                            "Try again", href="/system-2", refresh=True, className="mb-2 d-block"
                        ),
                        html.Div(
                            className="row row-cols-1 row-cols-5",
                            id="recommended-movies",
                        ),
                    ],
                    type="circle",
                ),
            ]
        )

@app.callback(Output("genre-output", "children"), Input("genre-dropdown", "value"))
def update_output(genre):
    if genre is None:
        return html.Div()
    else: 
        return [
            dbc.Row(
                [
                    html.Div(
                        [
                            *[
                                get_movie_card(movie)
                                for idx, movie in top_movies_in_genre(rating_merged, genre).iterrows()
                                    
                            ],
                        ],
                        className="row row-cols-1 row-cols-5",
                    ),
                ]
            ),
        ]


    
def get_movie_card(movie, with_rating=False):
    return html.Div(
        dbc.Card(
            [
                dbc.CardImg(
                    src=f"https://liangfgithub.github.io/MovieImages/{movie.movie_id}.jpg?raw=true",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        html.H6(movie.title, className="card-title text-center"),
                    ]
                ),
            ]
            + (
                [
                    dcc.RadioItems(
                        options=[
                            {"label": "1", "value": "1"},
                            {"label": "2", "value": "2"},
                            {"label": "3", "value": "3"},
                            {"label": "4", "value": "4"},
                            {"label": "5", "value": "5"},
                        ],
                        className="text-center",
                        id={"type": "movie_rating", "movie_id": movie.movie_id},
                        inputClassName="m-1",
                        labelClassName="px-1",
                    )
                ]
                if with_rating
                else []
            ),
            className="h-100",
        ),
        className="col mb-4",
    )
    
@app.callback(
    Output("rate-movie-container", "style"),
    Output("your-recommendation", "style"),
    [Input("button-recommend", "n_clicks")],
    prevent_initial_call=True,
)    
def on_recommend_button_clicked(n):
    return {"display": "none"}, {"display": "block"}

@app.callback(
    Output("recommended-movies", "children"),
    [Input("rate-movie-container", "style")],
    [
        State({"type": "movie_rating", "movie_id": ALL}, "value"),
        State({"type": "movie_rating", "movie_id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def on_getting_recommendations(style, ratings, ids):
    rating_input = np.full(100, np.nan)
    for i, rating in enumerate(ratings):
        if rating is not None:
            movie_id = ids[i]["movie_id"]
            if 0 <= movie_id < 100:
                rating_input[movie_id-1] = int(rating)           
    recommended_movies = myIBCF(rating_input, pruned_similarity_matrix, ratings_df)
    return [get_movie_card(movie) for idx, movie in recommended_movies.iterrows()]

@app.callback(
    Output("button-recommend", "disabled"),
    Input({"type": "movie_rating", "movie_id": ALL}, "value"),
)
def update_button_recommened_visibility(values):
    return not list(filter(None, values))

if __name__ == "__main__":
    app.run_server(port=8080, debug=True)