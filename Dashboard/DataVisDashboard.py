import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import date
import plotly.express as px
import pandas as pd
import folium
from streamlit_folium import folium_static
import pydeck as pdk
import json
from folium.plugins import MarkerCluster
import textwrap
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import folium
import networkx as nx
from streamlit_folium import folium_static
import time
import ast
import streamlit.components.v1 as components
import seaborn as sns
import pickle
import plotly.graph_objects as go
import json
import numpy as np


st.set_page_config(page_title="Data Visualization Project", layout="wide", page_icon="DALL·E-2024-05-04-22.05.ico")


st.sidebar.image('logo_luiss.png', width=150)
st.sidebar.write('''
## Navigate through the dashboard
''')
# Dropdown menu for navigation




    
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)
with st.sidebar:
    
    page = option_menu(
        "Choose a page", 
        ["Home Page", "Network", "Newspaper", "Birthplace Map", "Genre Dashboard"],
            icons=["house","","journal","map","table"], menu_icon='search')
    lottie = load_lottiefile("Animation - 1714854646071.json")
    st_lottie(lottie,key='loc')

if page == "Home Page":
    
    
    st.image('logo_luiss.png', width=305)

    # Display the second image in the second column
    
    st.write('''
    ## LUISS Guido Carli  
    ## Data Visualization - A.Y. 2023-2024  
    # Data Visualization course final project - WikiMusic2023  
   
    Coci Marco, Marchioni Gian Lorenzo, Filippo Navarra, Vincenzo Camerlengo
    ***
        

    
    ## Data Collection Process Overview

    In this data visualization project, we utilized Wikipedia's API and Wikidata to assemble a dataset focused on notable musicians, including the following attributes:
    - *Name*
    - *Musical Genre*
    - *Birthplace* (with coordinates)
    - *Nationality*
    - *Full article text*
    - *Links* embedded in the article that point to other Wikipedia articles
    - *Profile picture*
    - *Daily and yearly visits*

    Our selection criteria on Wikidata aimed to identify individuals and groups with roles such as singers, guitarists, songwriters, rappers, or 'Musical Group', all active since 1960 and having received at least one award. This targeted approach ensured we captured significant figures in the music industry.

    We then used Wikipedia to track the total views for each musician in 2023, narrowing our dataset down to the top 1000 most viewed musicians based on their popularity. This curated list formed the basis of our analysis and subsequent visualizations.
    ''')


   

    df = pd.read_csv("names_genres_text_visits_nationality_coordinates.csv")
    st.dataframe(df.head(15))
    

    def load_html(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content

    # Dictionary of file paths
    html_files = {
        "Sunburst Continent-Country-Genre": "continent-country-genre.html",
        "Sunburst Genre-Continent-Country": "genre-continent-country.html"
    }

    # Title for the app
    st.title('Interactive Sunburst Chart Viewer')

    # Select box for choosing the HTML file
    selected_chart = st.radio("Choose a chart to display:", list(html_files.keys()))

    # Load and display the selected HTML file
    html_content = load_html(html_files[selected_chart])
    components.html(html_content, height=1000)
            

if page == "Newspaper":

    st.write("""
    ## Artist of the Day
    In our daily creative visualization for the year 2023, we identified the artist who garnered the highest number of views each day.  
    We also present an excerpt from the introduction of the artist's Wikipedia article. Fluctuations in Wikipedia search volumes can often be attributed to events such as birthdays, deaths, or death anniversaries. This visualization makes it easy to investigate these correlations.
    Additionally, Wikidata queries facilitate the retrieval of thumbnail images for the articles, which enhances the visual appeal of our presentation.
             """)
    def load_data():
        return pd.read_csv("artists_introduction.csv")

    def overlay_text(base_image, text, position, font_path="arial.ttf", font_size=20, font_color="black", line_width=40,max_chars=620):
        try:
            # Load a font
            font = ImageFont.truetype(font_path, font_size)
            draw = ImageDraw.Draw(base_image)
            if len(text) > max_chars:
                text = text[:max_chars] + '...'  # Add ellipsis to indicate text is tr
            # Wrap the text
            wrapped_text = textwrap.fill(text, width=line_width)
            # Draw text
            draw.text(position, wrapped_text, font=font, fill=font_color)
        except Exception as e:
            st.error(f"Failed to overlay text: {e}")

    def overlay_image(base_image, image_path, position, size=(400, 400)):  # Default size can be adjusted
        try:
            artist_image = Image.open(image_path)
            artist_image = artist_image.resize(size, Image.LANCZOS)
            
            if artist_image.mode == 'RGBA':
                # Use alpha channel as a mask for transparency
                base_image.paste(artist_image, position, artist_image)
            else:
                base_image.paste(artist_image, position)
                
        except Exception as e:
            st.error(f"Failed to overlay image: {e}")

    def display_images_for_date(date, template):
        folder_path = "top1_artist_images_2"
        formatted_date = date.strftime('%Y-%m-%d')
        position = (70, 320)  # Position for top artist image
        artist_name = "Unknown"  # Default artist name if not found

        for filename in os.listdir(folder_path):
            if filename.startswith(formatted_date):
                image_path = os.path.join(folder_path, filename)
                overlay_image(template, image_path, position)
                artist_name = filename[len(formatted_date)+1:-4].replace('_', ' ')
                break

        return artist_name

    def main():
        st.title('Daily Artist News Page 📰')
        df = load_data()

        selected_date = st.date_input("Select a date", value=date(2023, 1, 1), min_value=date(2023, 1, 1), max_value=date(2023, 12, 31))
        template_path = "journal_image.png"
        newspaper_template = Image.open(template_path)

        artist_name = display_images_for_date(selected_date, newspaper_template)
        intro_text = df.loc[df['date'] == selected_date.strftime('%Y-%m-%d'), 'Introduction'].iloc[0] if not df.empty else "No introduction available."

        overlay_text(newspaper_template, intro_text, (500, 320))
        modified_path = f"temp_{selected_date.strftime('%Y-%m-%d')}.jpg"
        newspaper_template.save(modified_path)

        # Center the image and adjust width
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            st.image(modified_path, caption=f"Top artist on {selected_date.strftime('%Y-%m-%d')} is: {artist_name}", use_column_width=True)
        os.remove(modified_path)

    if __name__ == "__main__":
        main()






if page == "Home Page":

    


    st.title('Bar Race of Wikipedia Visits in 2023')
    

    # Display a video from a local file
    video_file = open("weeklydata, speed 700.mp4", 'rb')  # Open the video file in binary mode
    st.video(video_file)
    # Define a function that we will pass to the FastMarkerCluster




    

    



    
    
    def load_data():
        df = pd.read_csv('artist_birthplaces_with_coordinates.csv')
        link = pd.read_csv("names_links_exploded.csv")
        return df, link

    def create_graph(df, link):
        df = df.dropna(subset=['latitude', 'longitude'])
        G = nx.DiGraph()
        selected_artists = link['name'].drop_duplicates().tolist()
        coordinates = df[df['Artist'].isin(selected_artists)].set_index('Artist')[['latitude', 'longitude']].to_dict('index')
        
        for artist, coord in coordinates.items():
            G.add_node(artist, pos=(coord['longitude'], coord['latitude']))
        
        for _, row in link.iterrows():
            if row['name'] in selected_artists and row['links_from_article'] in selected_artists:
                if row['name'] in G.nodes and row['links_from_article'] in G.nodes:
                    G.add_edge(row['name'], row['links_from_article'])
        return G, df, selected_artists

    def plot_map(G, df, selected_artist=None):
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=3)
        
        # Define a set of offsets for markers that overlap
        offsets = [(0, 0), (0.0001, 0), (-0.0001, 0), (0, 0.0001), (0, -0.0001),
                (0.0001, 0.0001), (-0.0001, -0.0001), (0.0001, -0.0001), (-0.0001, 0.0001)]
        used_positions = {}  # To keep track of positions already used

        def get_offset_position(pos):
            """Apply offsets to position to avoid overlapping markers."""
            for dx, dy in offsets:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if new_pos not in used_positions:
                    used_positions[new_pos] = True
                    return new_pos
            return pos  # Return original if all offsets are used (unlikely)

        if selected_artist:
            print(f"Selected artist: {selected_artist}")
            if selected_artist in G:
                incoming_neighbors = set(G.predecessors(selected_artist))
                outgoing_neighbors = set(G.successors(selected_artist))
                relevant_nodes = set([selected_artist]) | incoming_neighbors | outgoing_neighbors
                print(f"Relevant nodes: {relevant_nodes}")

                # Plot all relevant nodes with offsets
                for node in relevant_nodes:
                    data = G.nodes[node]
                    pos = data['pos']
                    offset_pos = get_offset_position((pos[1], pos[0]))  # Applying offset to latitude and longitude
                    icon_color = 'green' if node == selected_artist else 'red'
                    tooltip = f"{node} (selected)" if node == selected_artist else node
                    popup_text = (f"Incoming connections are: {len(incoming_neighbors)} Outgoing connections are: {len(outgoing_neighbors)}") if node == selected_artist else ""
                                
                    popup = folium.Popup(popup_text, parse_html=True)
                    marker = folium.Marker(
                        location=[offset_pos[0], offset_pos[1]],  # Apply offset to marker position
                        tooltip=tooltip,
                        icon=folium.Icon(color=icon_color),
                        popup=popup if node == selected_artist else None
                    )
                    marker.add_to(m)

                # Plot incoming and outgoing edges with different colors
                for edge in G.edges():
                    if edge[1] == selected_artist or edge[0] == selected_artist:
                        start_pos = G.nodes[edge[0]]['pos']
                        end_pos = G.nodes[edge[1]]['pos']
                        start_offset_pos = get_offset_position((start_pos[1], start_pos[0]))
                        end_offset_pos = get_offset_position((end_pos[1], end_pos[0]))
                        line_color = 'blue' if edge[1] == selected_artist else 'orange'
                        folium.PolyLine(
                            locations=[[start_offset_pos[0], start_offset_pos[1]], [end_offset_pos[0], end_offset_pos[1]]],
                            color=line_color,
                            weight=3
                        ).add_to(m)
            else:
                print("Selected artist not in graph")
        else:
            print("No artist selected")
            # Plot all nodes and edges in a default color
            for node, data in G.nodes(data=True):
                pos = data['pos']
                offset_pos = get_offset_position((pos[1], pos[0]))
                marker = folium.Marker(
                    location=[offset_pos[0], offset_pos[1]],
                    tooltip=node,
                    icon=folium.Icon(color='blue')
                )
                marker.add_to(m)
            for edge in G.edges():
                start_pos = G.nodes[edge[0]]['pos']
                end_pos = G.nodes[edge[1]]['pos']
                start_offset_pos = get_offset_position((start_pos[1], start_pos[0]))
                end_offset_pos = get_offset_position((end_pos[1], end_pos[0]))
                folium.PolyLine(
                    locations=[[start_offset_pos[0], start_offset_pos[1]], [end_offset_pos[0], end_offset_pos[1]]],
                    color='blue',
                    weight=1
                ).add_to(m)


            

        return m








    def main():
        st.title('Artist Connections Map')
        df, link = load_data()
        G, df, artists = create_graph(df, link)

        artist_selected = st.selectbox('Select an Artist to Highlight:', options=['None'] + artists)
        
        if artist_selected == 'None':
            artist_selected = None

        m = plot_map(G, df, selected_artist=artist_selected)
        folium_static(m, width=1500, height=700)
        


        st.markdown("""
            <style>
            .red {color: red;}
            .green {color: green;}
            .blue {color: blue;}
            .orange {color: orange;}
            </style>
            The selected artist's marker is displayed in <span class="green">green</span>, while the markers of artists connected to it are displayed in <span class="red">red</span>. Incoming edges are shown in <span class="blue">blue</span> and outgoing edges in <span class="orange">orange</span>. By selecting the 'none' option, edges connecting all artists are displayed.
            """, unsafe_allow_html=True)

    if __name__ == "__main__":
        main()






if page == "Birthplace Map":


    

    def load_data():
        data = pd.read_csv("artist_birthplaces_with_coordinates.csv")
        return data.groupby(['Birthplace', 'latitude', 'longitude']).agg(
            Artist_Count=('Artist', 'size'),
            Artists=('Artist', lambda x: ', '.join(x))
        ).reset_index()


    def create_map(data):
        # Ensure there is data to process
        

        # Initialize the map centered around the mean latitude and longitude
        m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=2)
        
        # Create a MarkerCluster object
        marker_cluster = MarkerCluster().add_to(m)

        # Add markers to the cluster instead of the map
        for i in range(len(data)):
            folium.Marker(
                location=[data.iloc[i]['latitude'], data.iloc[i]['longitude']],
                popup=f"<b>Birthplace:</b> {data.iloc[i]['Birthplace']}<br><b>Artists:</b> {data.iloc[i]['Artists']}<br><b>Number of Artists:</b> {data.iloc[i]['Artist_Count']}",
                icon=folium.Icon(icon='user', prefix='glyphicon')
            ).add_to(marker_cluster)

        return m



        
    def main():
        st.title("Artists' Birthplaces Map 🌍")
        st.write(""" The map displayed below visualizes the birthplaces of artists, as sourced from Wikipedia data. 
             It is designed to cluster individuals born in geographically proximate areas, enhancing the visualization's clarity.
              As you zoom in, the map provides more granular details about each location. 
             Interactive markers on the map can be clicked to reveal a popup, which displays the number of artists born in that specific area along with the name of the city.
             """)

        # Custom CSS to shift the map slightly to the left
    
        data = load_data()
        map_folium = create_map(data)
        folium_static(map_folium, width=1500, height=700)  # Specify the size if needed

    if __name__ == "__main__":
        main()




if page == "Genre Dashboard":
        
        st.write('''
        ## Dashboard by Genre
        We organized our data by artist genre to facilitate some compelling visualizations. 
        These include a word cloud, where we trimmed the most frequently appearing words to emphasize the more distinctive ones. 
        Additionally, we created a line race visualization to display the cumulative daily views of the most visited artists within each genre. 
        We also analyzed the daily views for each of the top artists in their respective genres.
        ''')
        
        st.write("Pick your Favourite Genre:")
        genre_option = st.selectbox(
            "Select Genre",
            ["Pop", "Rock", "Hip Hop", "Country Music"],
            key='genre_dashboard_option'
        )
        st.header(f"Genre Dashboard - {genre_option}")

        if genre_option == "Pop":
            st.image("pop_1.jpg")
            st.caption("Word Cloud For Pop Music")
            @st.cache_data
            def load_data():
                df = pd.read_csv('views_genre.csv')
                df['date'] = pd.to_datetime(df['date'])
                return df[df['Genre'] == 'pop music']

            def filter_top_artists(pop):
                cumulative_views = pop.groupby('Artist')['views'].sum().reset_index()
                top_artists = cumulative_views.nlargest(5, 'views')['Artist']
                return pop[pop['Artist'].isin(top_artists)]

            def create_pop_plot(pop, current_date):
                if pop.empty:
                    st.error("No data available for Pop Music.")
                    return None

                pop_filtered = pop[pop['date'] <= pd.Timestamp(current_date)]
                pop_filtered = pop_filtered.copy()
                pop_filtered['cumulative_views'] = pop_filtered.groupby('Artist')['views'].cumsum()
                pop_filtered['cumulative_views_millions'] = pop_filtered['cumulative_views'] / 1e6

                fig = px.line(
                    pop_filtered,
                    x='date',
                    y='cumulative_views_millions',
                    color='Artist',
                    labels={'cumulative_views_millions': 'Cumulative Views (millions)'},
                    title=f'Cumulative Views by {current_date.strftime("%Y-%m-%d")} for Top Pop Music Artists'
                )
                return fig

            def main():
                st.title('Pop Music Popularity Over Time')
                pop = load_data()
                pop = filter_top_artists(pop)  # Filter for top artists

                if not pop.empty:
                    min_date = pop['date'].min()
                    max_date = pop['date'].max()
                    total_days = (max_date - min_date).days
                    plot_placeholder = st.empty()

                    for day in range(total_days + 1):
                        current_date = min_date + pd.DateOffset(days=day)
                        pop_fig = create_pop_plot(pop, current_date)
                        if pop_fig:
                            plot_placeholder.plotly_chart(pop_fig, use_container_width=True)
                        time.sleep(0.01)

            if __name__ == "__main__":
                main()


            

            

            @st.cache_data
            def load_data():
                df = pd.read_csv('views_genre.csv')
                df['date'] = pd.to_datetime(df['date'])
                return df[df['Genre'] == 'pop music']

            def prepare_and_plot(df, max_date):
                filtered_df = df[df['date'] <= max_date]
                top_artists = filtered_df.groupby('Artist')['views'].sum().nlargest(5).index
                top_artists_df = filtered_df[filtered_df['Artist'].isin(top_artists)]

                fig = px.line(top_artists_df, x='date', y='views', color='Artist',
                            title='Visits Over Time for the Top 5 Pop Music Artists',
                            labels={'date': 'Date', 'views': 'Number of Visits'})
                # Update layout to adjust figure size
                fig.update_layout({
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'width': 1400,  # Specify the width of the figure
                    'height': 700  # Specify the height of the figure
                })
                return fig

            df = load_data()
            max_date = df['date'].min()

            st.title('Visits in 2023')
            placeholder = st.empty()

            # Animation loop
            for _ in range((df['date'].max() - df['date'].min()).days + 1):
                max_date += pd.Timedelta(days=1)
                fig = prepare_and_plot(df, max_date)
                placeholder.plotly_chart(fig)
                time.sleep(0.1)     

            if __name__ == "__main__":
                main()
        


        if genre_option == "Rock":
            
            st.image("rock_1.jpg")
            st.caption("Word Cloud For Rock Music")

    
            def load_rock_data():
    # Load the complete dataset
                df = pd.read_csv('views_genre.csv')  # Adjust the path to your CSV file
                df['date'] = pd.to_datetime(df['date'])  # Convert to datetime without stripping time

                # Filter data for rock music genre only
                rock_music_df = df[df['Genre'] == 'rock music']
                return rock_music_df

            def filter_top_artists(rock_df):
                # Calculate cumulative views for each artist
                cumulative_views = rock_df.groupby('Artist')['views'].sum().reset_index()
                # Select top 5 artists based on cumulative views
                top_artists = cumulative_views.nlargest(5, 'views')['Artist']
                return rock_df[rock_df['Artist'].isin(top_artists)]

            def create_rock_plot(rock_df, current_date):
                if rock_df.empty:
                    st.error("No data available for Rock Music.")
                    return None

                # Filter data up to the selected date, ensuring both are Timestamp objects
                rock_df_filtered = rock_df[rock_df['date'] <= pd.Timestamp(current_date)]

                # Compute cumulative views by artist
                rock_df_filtered['cumulative_views'] = rock_df_filtered.groupby('Artist')['views'].cumsum()
                rock_df_filtered['cumulative_views_millions'] = rock_df_filtered['cumulative_views'] / 1e6

                # Create a Plotly line chart for rock music
                fig = px.line(
                    rock_df_filtered,
                    x='date',
                    y='cumulative_views_millions',
                    color='Artist',
                    labels={'cumulative_views_millions': 'Cumulative Views (millions)'},
                    title=f'Cumulative Views by {current_date.strftime("%Y-%m-%d")} for Top Rock Music Artists'
                )
                return fig

            def main():
                st.title('Rock Music Popularity Over Time')
                rock_df = load_rock_data()
                rock_df = filter_top_artists(rock_df)  # Filter for top artists

                if not rock_df.empty:
                    # Setup for dynamic plotting
                    min_date = rock_df['date'].min()
                    max_date = rock_df['date'].max()
                    total_days = (max_date - min_date).days
                    plot_placeholder = st.empty()

                    # Loop to update plot dynamically
                    for day in range(total_days + 1):
                        current_date = min_date + pd.DateOffset(days=day)
                        rock_fig = create_rock_plot(rock_df, current_date)
                        if rock_fig:
                            plot_placeholder.plotly_chart(rock_fig, use_container_width=True)
                        time.sleep(0.01)  # Adjust the sleep time to control the speed of the 'animation'

            if __name__ == "__main__":
                main()

                @st.cache_data
                def load_data():
                    df = pd.read_csv('views_genre.csv')
                    df['date'] = pd.to_datetime(df['date'])
                    return df[df['Genre'] == 'rock music'] 

                def prepare_and_plot(df, max_date):
                    filtered_df = df[df['date'] <= max_date]
                    top_artists = filtered_df.groupby('Artist')['views'].sum().nlargest(5).index
                    top_artists_df = filtered_df[filtered_df['Artist'].isin(top_artists)]

                    fig = px.line(top_artists_df, x='date', y='views', color='Artist',
                                title='Visits Over Time for the Top 5 Rock Music Artists',
                                labels={'date': 'Date', 'views': 'Number of Visits'})
                    fig.update_layout({
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'plot_bgcolor': 'rgba(0,0,0,0)',
                        'width': 1400,
                        'height': 700
                    })
                    return fig

                df = load_data()
                max_date = df['date'].min()

                st.title('Visits in 2023 - Rock Music')
                placeholder = st.empty()

                # Animation loop
                for _ in range((df['date'].max() - df['date'].min()).days + 1):
                    max_date += pd.Timedelta(days=1)
                    fig = prepare_and_plot(df, max_date)
                    placeholder.plotly_chart(fig)
                    time.sleep(0.1)
                
                if __name__ == "__main__":
                    main()

                

            

            

            
                
        

        if genre_option == "Hip Hop":

            st.image("hip_1.jpg")
            st.caption("Word Cloud For Hip Hop Music")

            def load_hip_hop_data():
                # Load the complete dataset
                df = pd.read_csv('views_genre.csv')  # Adjust the path to your CSV file
                df['date'] = pd.to_datetime(df['date'])  # Convert to datetime without stripping time

                # Filter data for hip hop music genre only
                hip_hop = df[df['Genre'] == 'hip hop music']
                return hip_hop

            def filter_top_artists(hip_hop):
                # Calculate cumulative views for each artist
                cumulative_views = hip_hop.groupby('Artist')['views'].sum().reset_index()
                # Select top 5 artists based on cumulative views
                top_artists = cumulative_views.nlargest(5, 'views')['Artist']
                return hip_hop[hip_hop['Artist'].isin(top_artists)]

            def create_hip_hop_plot(hip_hop, current_date):
                if hip_hop.empty:
                    st.error("No data available for Hip Hop Music.")
                    return None

                # Filter data up to the selected date, ensuring both are Timestamp objects
                hip_hop_filtered = hip_hop[hip_hop['date'] <= pd.Timestamp(current_date)]

                # Compute cumulative views by artist
                hip_hop_filtered['cumulative_views'] = hip_hop_filtered.groupby('Artist')['views'].cumsum()
                hip_hop_filtered['cumulative_views_millions'] = hip_hop_filtered['cumulative_views'] / 1e6

                # Create a Plotly line chart for hip hop music
                fig = px.line(
                    hip_hop_filtered,
                    x='date',
                    y='cumulative_views_millions',
                    color='Artist',
                    labels={'cumulative_views_millions': 'Cumulative Views (millions)'},
                    title=f'Cumulative Views by {current_date.strftime("%Y-%m-%d")} for Top Hip Hop Music Artists'
                )
                return fig

            def main():
                st.title('Hip Hop Music Popularity Over Time')
                hip_hop = load_hip_hop_data()
                hip_hop = filter_top_artists(hip_hop)  # Filter for top artists

                if not hip_hop.empty:
                    # Setup for dynamic plotting
                    min_date = hip_hop['date'].min()
                    max_date = hip_hop['date'].max()
                    total_days = (max_date - min_date).days
                    plot_placeholder = st.empty()

                    # Loop to update plot dynamically
                    for day in range(total_days + 1):
                        current_date = min_date + pd.DateOffset(days=day)
                        hip_hop_fig = create_hip_hop_plot(hip_hop, current_date)
                        if hip_hop_fig:
                            plot_placeholder.plotly_chart(hip_hop_fig, use_container_width=True)
                        time.sleep(0.01)  # Adjust the sleep time to control the speed of the 'animation'

            if __name__ == "__main__":
                main()




            @st.cache_data
            def load_data():
                df = pd.read_csv('views_genre.csv')
                df['date'] = pd.to_datetime(df['date'])
                return df[df['Genre'] == 'hip hop music']

            def prepare_and_plot(df, max_date):
                filtered_df = df[df['date'] <= max_date]
                top_artists = filtered_df.groupby('Artist')['views'].sum().nlargest(5).index
                top_artists_df = filtered_df[filtered_df['Artist'].isin(top_artists)]

                fig = px.line(top_artists_df, x='date', y='views', color='Artist',
                            title='Visits Over Time for the Top 5 Hip Hop Music Artists',
                            labels={'date': 'Date', 'views': 'Number of Visits'})
                fig.update_layout({
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'width': 1400,
                    'height': 700
                })
                return fig

            df = load_data()
            max_date = df['date'].min()

            st.title('Visits in 2023 - Hip Hop Music')
            placeholder = st.empty()

            # Animation loop
            for _ in range((df['date'].max() - df['date'].min()).days + 1):
                max_date += pd.Timedelta(days=1)
                fig = prepare_and_plot(df, max_date)
                placeholder.plotly_chart(fig)
                time.sleep(0.1)
            

            if __name__ == "__main__":
                main()
                
        

     


        if genre_option == "Country Music":
                
                st.image("country_1.jpg")
                st.caption("Word Cloud For Country Music")
                def load_country_data():
    # Load the complete dataset
                    df = pd.read_csv('views_genre.csv')  # Adjust the path to your CSV file
                    df['date'] = pd.to_datetime(df['date'])  # Convert to datetime without stripping time

                    # Filter data for country music genre only
                    country = df[df['Genre'] == 'country music']
                    return country

                def filter_top_artists(country):
                    # Calculate cumulative views for each artist
                    cumulative_views = country.groupby('Artist')['views'].sum().reset_index()
                    # Select top 5 artists based on cumulative views
                    top_artists = cumulative_views.nlargest(5, 'views')['Artist']
                    return country[country['Artist'].isin(top_artists)]

                def create_country_plot(country, current_date):
                    if country.empty:
                        st.error("No data available for Country Music.")
                        return None

                    # Filter data up to the selected date, ensuring both are Timestamp objects
                    country_filtered = country[country['date'] <= pd.Timestamp(current_date)]

                    # Compute cumulative views by artist
                    country_filtered['cumulative_views'] = country_filtered.groupby('Artist')['views'].cumsum()
                    country_filtered['cumulative_views_millions'] = country_filtered['cumulative_views'] / 1e6

                    # Create a Plotly line chart for country music
                    fig = px.line(
                        country_filtered,
                        x='date',
                        y='cumulative_views_millions',
                        color='Artist',
                        labels={'cumulative_views_millions': 'Cumulative Views (millions)'},
                        title=f'Cumulative Views by {current_date.strftime("%Y-%m-%d")} for Top Country Music Artists'
                    )
                    return fig

                def main():
                    st.title('Country Music Popularity Over Time')
                    country = load_country_data()
                    country = filter_top_artists(country)  # Filter for top artists

                    if not country.empty:
                        # Setup for dynamic plotting
                        min_date = country['date'].min()
                        max_date = country['date'].max()
                        total_days = (max_date - min_date).days
                        plot_placeholder = st.empty()

                        # Loop to update plot dynamically
                        for day in range(total_days + 1):
                            current_date = min_date + pd.DateOffset(days=day)
                            country_fig = create_country_plot(country, current_date)
                            if country_fig:
                                plot_placeholder.plotly_chart(country_fig, use_container_width=True)
                            time.sleep(0.01)  # Adjust the sleep time to control the speed of the 'animation'
                if __name__ == "__main__":
                    main()


                @st.cache_data
                def load_data():
                    df = pd.read_csv('views_genre.csv')
                    df['date'] = pd.to_datetime(df['date'])
                    return df[df['Genre'] == 'country music']

                def prepare_and_plot(df, max_date):
                    filtered_df = df[df['date'] <= max_date]
                    top_artists = filtered_df.groupby('Artist')['views'].sum().nlargest(5).index
                    top_artists_df = filtered_df[filtered_df['Artist'].isin(top_artists)]

                    fig = px.line(top_artists_df, x='date', y='views', color='Artist',
                                title='Visits Over Time for the Top 5 Country Music Artists',
                                labels={'date': 'Date', 'views': 'Number of Visits'})
                    fig.update_layout({
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'plot_bgcolor': 'rgba(0,0,0,0)',
                        'width': 1400,
                        'height': 700
                    })
                    return fig

                df = load_data()
                max_date = df['date'].min()

                st.title('Visits in 2023 - Country Music')
                placeholder = st.empty()

                # Animation loop
                for _ in range((df['date'].max() - df['date'].min()).days + 1):
                    max_date += pd.Timedelta(days=1)
                    fig = prepare_and_plot(df, max_date)
                    placeholder.plotly_chart(fig)
                    time.sleep(0.1)

                if __name__ == "__main__":
                    main()
            
                
        
        

            
if page == "Network":




    genres_df=pd.read_csv('names_genres_for_network.csv')
    genres_dict = genres_df.set_index('Artist')['Genre'].to_dict()  

    with open('layout.json', 'r') as f:
        loaded_layout = json.load(f)
        x_nodes, y_nodes = zip(*loaded_layout)

    with open('graph.pickle', 'rb') as f:
        graph = pickle.load(f)

    genre_colors = {
        'Unspecified': 'gray',
        'Rock': 'red',
        'Pop': 'green',
        'Hip Hop': 'blue',
        'Country': 'yellow',
        'Jazz and Blues': 'purple',
        'Other': 'orange'
    }

    # Extract node attributes for hover information
    node_names = graph.vs['name']
    node_communities = graph.vs['community']
    node_visits = graph.vs['visits']
    node_sizes = [max(np.log10(visit) * 3, 0.2) for visit in node_visits]  # Adjust node size based on visits
    node_degree = graph.vs["degree"]
    node_betweenness = graph.vs["betweenness"] 
    node_indegree = [graph.degree(v.index, mode="IN") for v in graph.vs]  # Get indegree for all vertices

    # Extract the genre for each node from the dictionary
    node_genres = [genres_dict.get(name, 'Unspecified') for name in node_names]

    # Generate edge traces for Plotly
    edge_x, edge_y = [], []
    for edge in graph.es:
        source = edge.source
        target = edge.target
        edge_x.extend([x_nodes[source], x_nodes[target], None])
        edge_y.extend([y_nodes[source], y_nodes[target], None])

    # Define a color map for genres
    genre_colors = {
        'Unspecified': 'gray',
        'Rock': 'red',
        'Pop': 'green',
        'Hip Hop': 'blue',
        'Country': 'yellow',
        'Jazz and Blues': 'purple',
        'Other': 'orange'
    }

    # Define a color map for communities
    num_communities = len(set(node_communities))
    colors = [f'hsl({(i / num_communities) * 360}, 70%, 60%)' for i in range(num_communities)]

    st.write('''
    ## Network Analysis Overview  
    This analysis explores the interconnections among 1,000 artists listed in our dataset derived from Wikipedia links. 
    We specifically looked at hyperlinks that connect one artist's page to another within our set, forming a directed graph that illustrates relationships and influence among musicians.

    We applied the **Leiden algorithm** to detect communities within this network.  
    Artists were also categorized by their primary musical genre, allowing for a different grouping:
    - **Unspecified**: Artists without a clear genre classification or whose data was incomplete.
    - **Five most popular genres**: Each representing a significant portion of our dataset.
    - **Other**: Encompassing all remaining genres.

    The visualizations provided offer insights into the structure of these relationships, highlighting influential artists and the clusters or communities they form. 
    Interactive features allow for toggling edges, viewing and hiding communities, and adjusting node coloring based on community detection or genre classification.
    ''')

    toggle_edges=st.radio('Toggle Edges', ('View Edges', 'Hide Edges'))
    toggle_color=st.radio('Node Color', ('Community', 'Genre'))
        
    if toggle_color=='Community' and toggle_edges=='View Edges':

        # Create a separate trace for each community
        fig = go.Figure()

        for i in range(num_communities):
            # Select nodes in this community
            indices = [index for index, val in enumerate(node_communities) if val == i]
            community_x = [x_nodes[i] for i in indices]
            community_y = [y_nodes[i] for i in indices]
            community_names = [node_names[i] for i in indices]
            community_visits = [node_visits[i] for i in indices]
            community_indegree = [node_indegree[i] for i in indices]
            community_betweenness = [node_betweenness[i] for i in indices]
            community_degree = [node_degree[i] for i in indices]

            # Node sizes based on indegree, scaled logarithmically
            community_sizes = [max(np.log10(indegree + 1) * 10, 5) for indegree in community_indegree]

            # Create trace for each community
            fig.add_trace(go.Scatter(
                x=community_x, y=community_y,
                mode='markers',
                marker=dict(
                    size=community_sizes,
                    color=colors[i],  # Assume this is predefined or calculated elsewhere
                    line=dict(color='black', width=1)
                ),
                name=f'Community {i}',  # Legend entry
                hoverinfo='text',
                hovertext=[
                    f'Name: {name}<br>Community: {i}<br>Visits: {visits}<br>In-degree: {indegree}<br>'
                    f'Degree: {degree}<br>Betweenness: {round(betweenness, 2)}'
                    for name, visits, indegree, degree, betweenness in zip(
                        community_names, community_visits, community_indegree, community_degree, community_betweenness
                    )
                ]
            ))

        # Edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=3, color='rgba(150, 150, 150, 0.01)'),  # Edge properties
            hoverinfo='none',
            mode='lines',
            showlegend=False  # Do not show edge trace in the legend
        )

        # Add edge trace to the figure
        fig.add_trace(edge_trace)

        # Update layout
        fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            hovermode='closest',
            legend=dict(title="Communities", x=0, y=1, bgcolor='rgba(255,255,255,0.5)'),
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        # Show the figure
        st.plotly_chart(fig, use_column_width=True)


    if toggle_color=='Community' and toggle_edges=='Hide Edges':
        
        # Create a separate trace for each community
        fig = go.Figure()

        for i in range(num_communities):
            # Select nodes in this community
            indices = [index for index, val in enumerate(node_communities) if val == i]
            community_x = [x_nodes[i] for i in indices]
            community_y = [y_nodes[i] for i in indices]
            community_names = [node_names[i] for i in indices]
            community_visits = [node_visits[i] for i in indices]
            community_indegree = [node_indegree[i] for i in indices]
            community_betweenness = [node_betweenness[i] for i in indices]
            community_degree = [node_degree[i] for i in indices]

            # Node sizes based on indegree, scaled logarithmically
            community_sizes = [max(np.log10(indegree + 1) * 10, 5) for indegree in community_indegree]

            # Create trace for each community
            fig.add_trace(go.Scatter(
                x=community_x, y=community_y,
                mode='markers',
                marker=dict(
                    size=community_sizes,
                    color=colors[i],  # Assume this is predefined or calculated elsewhere
                    line=dict(color='black', width=1)
                ),
                name=f'Community {i}',  # Legend entry
                hoverinfo='text',
                hovertext=[
                    f'Name: {name}<br>Community: {i}<br>Visits: {visits}<br>In-degree: {indegree}<br>'
                    f'Degree: {degree}<br>Betweenness: {round(betweenness, 2)}'
                    for name, visits, indegree, degree, betweenness in zip(
                        community_names, community_visits, community_indegree, community_degree, community_betweenness
                    )
                ]
            ))

        # Update layout
        fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            hovermode='closest',
            legend=dict(title="Communities", x=0, y=1, bgcolor='rgba(255,255,255,0.5)'),
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        # Show the figure
        st.plotly_chart(fig, use_column_width=True)

    if toggle_color=='Genre' and toggle_edges=='View Edges':
        # Create a figure
        fig_genre = go.Figure()

        for genre, color in genre_colors.items():
            indices = [i for i, g in enumerate(node_genres) if g == genre]
            genre_x = [x_nodes[i] for i in indices]
            genre_y = [y_nodes[i] for i in indices]
            genre_names = [node_names[i] for i in indices]
            genre_visits = [node_visits[i] for i in indices]
            genre_indegree = [node_indegree[i] for i in indices]
            genre_degree = [node_degree[i] for i in indices]
            genre_betweenness = [node_betweenness[i] for i in indices]

            # Node sizes based on indegree, scaled logarithmically
            genre_sizes = [max(np.log10(indegree + 1) * 10, 5) for indegree in genre_indegree]

            # Create trace for this genre
            fig_genre.add_trace(go.Scatter(
                x=genre_x, y=genre_y,
                mode='markers',
                marker=dict(
                    size=genre_sizes,
                    color=color,
                    line=dict(color='black', width=1)
                ),
                name=genre,  # Name used in the legend
                hoverinfo='text',
                hovertext=[
                    f'Name: {name}<br>Genre: {genre}<br>Visits: {visits}<br>'
                    f'In-degree: {indegree}<br>Degree: {degree}<br>Betweenness: {round(betweenness, 2)}'
                    for name, visits, indegree, degree, betweenness in zip(
                        genre_names, genre_visits, genre_indegree, genre_degree, genre_betweenness
                    )
                ]
            ))
            
        # Edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=3, color='rgba(150, 150, 150, 0.01)'),  # Edge properties
            hoverinfo='none',
            mode='lines',
            showlegend=False  # Do not show edge trace in the legend
        )

        # Add edge trace to the figure
        fig_genre.add_trace(edge_trace)

        # Update the layout to adjust visual aesthetics
        fig_genre.update_layout(
            title="Node Attributes by Genre",
            paper_bgcolor='black',  # Black background color
            plot_bgcolor='black',  # Black plot background color
            hovermode='closest',  # Hover mode for closest point
            legend=dict(title="Genres", x=0, y=1, bgcolor='rgba(255,255,255,0.5)'),  # Legend with a semi-transparent background
            margin=dict(b=0, l=0, r=0, t=0),  # Minimal margins
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Clean x-axis
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)  # Clean y-axis
        )

        # Display the figure
        st.plotly_chart(fig_genre, use_column_width=True)

    if toggle_color=='Genre' and toggle_edges=='Hide Edges':
        # Create a figure
        fig_genre = go.Figure()

        for genre, color in genre_colors.items():
            indices = [i for i, g in enumerate(node_genres) if g == genre]
            genre_x = [x_nodes[i] for i in indices]
            genre_y = [y_nodes[i] for i in indices]
            genre_names = [node_names[i] for i in indices]
            genre_visits = [node_visits[i] for i in indices]
            genre_indegree = [node_indegree[i] for i in indices]
            genre_degree = [node_degree[i] for i in indices]
            genre_betweenness = [node_betweenness[i] for i in indices]

            # Node sizes based on indegree, scaled logarithmically
            genre_sizes = [max(np.log10(indegree + 1) * 10, 5) for indegree in genre_indegree]

            # Create trace for this genre
            fig_genre.add_trace(go.Scatter(
                x=genre_x, y=genre_y,
                mode='markers',
                marker=dict(
                    size=genre_sizes,
                    color=color,
                    line=dict(color='black', width=1)
                ),
                name=genre,  # Name used in the legend
                hoverinfo='text',
                hovertext=[
                    f'Name: {name}<br>Genre: {genre}<br>Visits: {visits}<br>'
                    f'In-degree: {indegree}<br>Degree: {degree}<br>Betweenness: {round(betweenness, 2)}'
                    for name, visits, indegree, degree, betweenness in zip(
                        genre_names, genre_visits, genre_indegree, genre_degree, genre_betweenness
                    )
                ]
            ))

        # Update the layout to adjust visual aesthetics
        fig_genre.update_layout(
            title="Node Attributes by Genre",
            paper_bgcolor='black',  # Black background color
            plot_bgcolor='black',  # Black plot background color
            hovermode='closest',  # Hover mode for closest point
            legend=dict(title="Genres", x=0, y=1, bgcolor='rgba(255,255,255,0.5)'),  # Legend with a semi-transparent background
            margin=dict(b=0, l=0, r=0, t=0),  # Minimal margins
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Clean x-axis
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)  # Clean y-axis
        )

        # Display the figure
        st.plotly_chart(fig_genre, use_column_width=True)


    st.write('''
    ***
    ## Network Metrics Insight

    The scatterplot below plots each artist's **Indegree** against their **Betweenness Centrality**:
    - **Indegree**: Measures the number of direct links directed towards an artist, indicating popularity or influence within the network.
    - **Betweenness Centrality**: Captures an artist's role as a bridge between different clusters. A high betweenness centrality indicates that the artist connects various groups, facilitating flow within the network, and often holding substantial strategic importance.

    This visualization helps to quickly identify key influencers based on their structural positions and connectivity in the graph.
    ''')


    fig_scatter = go.Figure()

    # Loop through each unique genre to create a trace
    for genre, color in genre_colors.items():
        # Get indices for nodes of this genre
        indices = [i for i, g in enumerate(node_genres) if g == genre]
        # Create a trace for each genre
        fig_scatter.add_trace(go.Scatter(
            x=[node_indegree[i] for i in indices],
            y=[node_betweenness[i] for i in indices],
            mode='markers',
            marker=dict(color=color, size=10),  # Adjust size as needed
            name=genre,  # This name appears in the legend
            text=[node_names[i] for i in indices],
            hoverinfo='text'
        ))

    # Add titles and adjust background colors
    fig_scatter.update_layout(
        title="Node Indegree vs Betweenness Centrality",
        xaxis_title="Indegree",
        yaxis_title="Betweenness Centrality",
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend_title="Genre",
        legend=dict(x=1, y=1, bgcolor='rgba(255,255,255,0.5)', bordercolor='Black')
    )

    st.plotly_chart(fig_scatter, use_column_width=True)

    st.write('''
    ***
    ## Genre and Community Correlations

    **Leiden Algorithm**: Building upon the foundations of the Louvain method, the Leiden algorithm addresses significant flaws in community detection. Unlike the Louvain method, which can sometimes produce disconnected or loosely connected communities, the Leiden algorithm ensures that all identified communities are well-connected and cohesive. This is a critical improvement, as our analysis indicates that with the Louvain method, up to 25% of the communities can be poorly connected, and about 16% might end up disconnected when the algorithm is applied iteratively.

    Moreover, the Leiden algorithm guarantees that, through iterative applications, it converges to a partition where every subset within a community is optimally clustered. This means that not only are communities guaranteed to be connected, each is internally structured to maximize connectivity and minimize inter-community links.

    **Performance and Efficiency**: The Leiden algorithm also surpasses the Louvain method in terms of computational efficiency. Utilizing a rapid local move technique, it operates faster and achieves better partitions compared to its predecessor, making it ideal for large and complex networks like ours.

    The heatmap below illustrates how musical genres intersect with these robustly defined communities. Each cell in the heatmap quantifies the number of artists from a particular genre within each detected community, shedding light on the distribution of musical styles across intricate network clusters.
    ''')

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('communities_genres_heatmap.png', width=600)
