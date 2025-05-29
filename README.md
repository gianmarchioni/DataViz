# Data Visualization Project Submission

Esteemed professors, this is our submission for the data visualization project.

## Data Compilation
We have compiled numerous notebooks and datasets, as we gathered our own dataset using Wikipedia's API and by querying Wikidata.
We believe the most comprehensive and intuitive way to explore our project is through the dashboard. All data and files required to run it are in the 'Dashboard' directory.

## Project Theme
The main theme involved collecting data on famous musicians from Wikipedia, focusing on the following attributes:
- **Name**
- **Musical Genre**
- **Birthplace** (with coordinates)
- **Nationality**
- **Full article text**
- **Links embedded in the article** that point to other Wikipedia articles
- **Profile picture**
- **Daily and yearly visits**

## Querying Process
We retrieved names from Wikidata under these conditions:
- **Occupation:** Musician, Singer, Guitarist, Songwriter, Rapper, or Entity is 'Musical Group'
- Filtered for musicians or groups active since 1960 and with at least one award

We then queried Wikipedia to obtain the total 2023 views for each and ultimately selected just the top 1000 names. This process is detailed in the notebook titled `wikipedia_api.ipynb`.

## Data Files
- `top_visited_2023.csv` contains the final list of names we selected, along with their total visits during 2023.
- 'NUMBER_of_views.csv' and 'top_visits_per_day.csv' contain daily views and the list of most viewed daily artists in 2023.
- 'names_links_exploded.csv' is just a list on links in tidy format, useful to create the network.
- 'names_genres_text_visits_nationality_coordinates.csv' is a file that comprises almost all the collected data, excluding links, images and daily views.
- 'artists_introduction.csv' stores snippets from the article headers, to be used in the newspaper visualization.
- 'views_genre.csv' contains daily views categorized by genre.
- 'artist_birthplaces_with_coordinates.csv' is redudant, but used for the dashboard.


Notebooks, such as `nationality_query.ipynb`, begin with the 1000 name list and query for more data.
'network_analysis.ipynb' fetches article links, filters them, and finally creates the network. 
'worldcloud_sunburst.ipynb' does nome visualizations based on 'names_genres_text_visits_nationality_coordinates.csv'.
'query_wikipedia.ipynb' contains all the requests to wikipedia in order to get needed information, such as genres or daily visits.
'bar_race_chart.ipynb' stores the code for the implementation of the bar race.
