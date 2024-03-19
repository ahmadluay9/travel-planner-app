# import library
import streamlit as st
import pandas as pd
import json
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import folium
from streamlit_folium import folium_static

import tiktoken
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
# api_key = os.getenv('API_KEY')
# openai_api_key = os.getenv("MY_OPENAI_KEY")
url = 'https://places.googleapis.com/v1/places:searchText'

# if not api_key:
#     raise ValueError("API_KEY not found in environment variables. Please set it in the .env file.")
# if not openai_api_key:
#     raise ValueError("MY_OPENAI_KEY not found in environment variables. Please set it in the .env file.")

def main():
    st.sidebar.title("Travel Recommendation App Demo")

    api_key = st.sidebar.text_input("Enter Google Maps API key:",type="password")
    openai_api_key = st.sidebar.text_input("Enter OpenAI API key:",type="password")

    st.sidebar.write('Please fill in the fields below.')
    destination = st.sidebar.text_input('Destination:',key='destination_app')
    min_rating = st.sidebar.number_input('Minimum Rating:',value=4.0,min_value=0.5,max_value=4.5,step=0.5,key='minrating_app')
    radius = st.sidebar.number_input('Search Radius in meter:',value=3000,min_value=500,max_value=50000,step=100,key='radius_app')
    
    if destination:
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': 'places.location',
            }
        data = {
            'textQuery': destination,
            'maxResultCount': 1,
        }

        # Convert data to JSON format
        json_data = json.dumps(data)

        # Make the POST request
        response = requests.post(url, data=json_data, headers=headers)

        # Print the response
        result = response.json()

        # Convert JSON data to DataFrame
        df = pd.json_normalize(result['places'])
        
        # Get the latitude and longitude values
        initial_latitude = df['location.latitude'].iloc[0]
        initial_longitude = df['location.longitude'].iloc[0]

        # Create the circle
        circle_center = {"latitude": initial_latitude, "longitude": initial_longitude}
        circle_radius = radius
        
        headers_place = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.priceLevel,places.userRatingCount,places.rating,places.websiteUri,places.location,places.googleMapsUri',
        }

        def hotel(): 
            data_hotel = {
                'textQuery': f'Place to stay near {destination}',
                'minRating': min_rating,
                'locationBias': {
                    "circle": {
                        "center": circle_center,
                        "radius": circle_radius
                    }
                }
            }

            # Convert data to JSON format
            json_data_hotel = json.dumps(data_hotel)  
            # Make the POST request
            response_hotel = requests.post(url, data=json_data_hotel, headers=headers_place)
            # Print the response
            result_hotel = response_hotel.json()
            # Convert JSON data to DataFrame
            df_hotel = pd.json_normalize(result_hotel['places'])
            # Add 'type'
            df_hotel['type'] = 'Hotel'
            return df_hotel
    
        def restaurant():  
            data_restaurant = {
                'textQuery': f'Place to eat near {destination}',
                'minRating': min_rating,
                'locationBias': {
                    "circle": {
                        "center": circle_center,
                        "radius": circle_radius
                    }
                }
            }

            # Convert data to JSON format
            json_data_restaurant = json.dumps(data_restaurant)  
            # Make the POST request
            response_restaurant = requests.post(url, data=json_data_restaurant, headers=headers_place)
            # Print the response
            result_restaurant = response_restaurant.json()
            # Convert JSON data to DataFrame
            df_restaurant = pd.json_normalize(result_restaurant['places'])
            # Add 'type'
            df_restaurant['type'] = 'Restaurant'
            return df_restaurant
    
        def tourist():  
            data_tourist = {
                'textQuery': f'Tourist attraction near {destination}',
                'minRating': min_rating,
                'locationBias': {
                    "circle": {
                        "center": circle_center,
                        "radius": circle_radius
                    }
                }
            }

            # Convert data to JSON format
            json_data_tourist = json.dumps(data_tourist)  
            # Make the POST request
            response_tourist = requests.post(url, data=json_data_tourist, headers=headers_place)
            # Print the response
            result_tourist = response_tourist.json()
            # Convert JSON data to DataFrame
            df_tourist = pd.json_normalize(result_tourist['places'])
            # Add 'type'
            df_tourist['type'] = 'Tourist'
            return df_tourist
    
        df_hotel = hotel()
        df_restaurant = restaurant()
        df_tourist = tourist()

    # Assuming all three dataframes have similar columns
        df_place = pd.concat([df_hotel, df_restaurant, df_tourist], ignore_index=True)
        df_place = df_place.sort_values(by=['userRatingCount', 'rating'], ascending=[False, False]).reset_index(drop=True)
        
        df_place_rename = df_place[['type','displayName.text','formattedAddress','rating', 'userRatingCount','googleMapsUri', 'websiteUri', 'location.latitude', 'location.longitude', 'displayName.languageCode']]
        df_place_rename = df_place_rename.rename(columns={
            'displayName.text': 'Name',
            'rating': 'Rating',
            'googleMapsUri': 'Google Maps URL',
            'websiteUri': 'Website URL',
            'userRatingCount': 'User Rating Count',
            'location.latitude': 'Latitude',
            'location.longitude': 'Longitude',
            'formattedAddress': 'Address',
            'displayName.languageCode': 'Language Code',
            'type': 'Type'
        })
        def database():
            st.dataframe(df_place_rename)

        def maps():
            st.header("🌏 Travel Recommendation App 🌏")

            places_type = st.radio('Looking for: ',["Hotels 🏨", "Restaurants 🍴","Tourist Attractions ⭐"])
            initial_location = [initial_latitude, initial_longitude]
            type_colour = {'Hotel':'blue', 'Restaurant':'green', 'Tourist':'orange'}
            type_icon = {'Hotel':'home', 'Restaurant':'cutlery', 'Tourist':'star'}

            st.write(f"# Here are our recommendations for {places_type} near {destination} ")

            if places_type == 'Hotels 🏨': 
                df_place = df_hotel
                with st.spinner("Just a moment..."):
                    for index,row in df_place.iterrows():
                        location = [row['location.latitude'], row['location.longitude']]
                        mymap  = folium.Map(location = initial_location, 
                                zoom_start=9, control_scale=True)
                        content = (str(row['displayName.text']) + '<br>' + 
                                'Rating: '+ str(row['rating']) + '<br>' + 
                                'Address: ' + str(row['formattedAddress']) + '<br>' + 
                                'Website: '  + str(row['websiteUri'])
                                )
                        iframe = folium.IFrame(content, width=300, height=125)
                        popup = folium.Popup(iframe, max_width=300)

                        icon_color = type_colour[row['type']]
                        icon_type = type_icon[row['type']]
                        icon = folium.Icon(color=icon_color, icon=icon_type)

                        # Use different icons for hotels, restaurants, and tourist attractions
                        folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)

                        st.write(f"## {index + 1}. {row['displayName.text']}")
                        folium_static(mymap)
                        st.write(f"Rating: {row['rating']}")
                        st.write(f"Address: {row['formattedAddress']}")
                        st.write(f"Website: {row['websiteUri']}")
                        st.write(f"More information: {row['googleMapsUri']}\n")
                            
            elif places_type == 'Restaurants 🍴': 
                df_place = df_restaurant
                with st.spinner("Just a moment..."):
                    for index,row in df_place.iterrows():
                        location = [row['location.latitude'], row['location.longitude']]
                        mymap  = folium.Map(location = initial_location, 
                                zoom_start=9, control_scale=True)
                        content = (str(row['displayName.text']) + '<br>' + 
                                'Rating: '+ str(row['rating']) + '<br>' + 
                                'Address: ' + str(row['formattedAddress']) + '<br>' + 
                                'Website: '  + str(row['websiteUri'])
                                )
                        iframe = folium.IFrame(content, width=300, height=125)
                        popup = folium.Popup(iframe, max_width=300)

                        icon_color = type_colour[row['type']]
                        icon_type = type_icon[row['type']]
                        icon = folium.Icon(color=icon_color, icon=icon_type)

                        # Use different icons for hotels, restaurants, and tourist attractions
                        folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)

                        st.write(f"## {index + 1}. {row['displayName.text']}")
                        folium_static(mymap)
                        st.write(f"Rating: {row['rating']}")
                        st.write(f"Address: {row['formattedAddress']}")
                        st.write(f"Website: {row['websiteUri']}")
                        st.write(f"More information: {row['googleMapsUri']}\n")
            else:
                df_place = df_tourist
                with st.spinner("Just a moment..."):
                    for index,row in df_place.iterrows():
                        location = [row['location.latitude'], row['location.longitude']]
                        mymap  = folium.Map(location = initial_location, 
                                zoom_start=9, control_scale=True)
                        content = (str(row['displayName.text']) + '<br>' + 
                                'Rating: '+ str(row['rating']) + '<br>' + 
                                'Address: ' + str(row['formattedAddress']) + '<br>' + 
                                'Website: '  + str(row['websiteUri'])
                                )
                        iframe = folium.IFrame(content, width=300, height=125)
                        popup = folium.Popup(iframe, max_width=300)

                        icon_color = type_colour[row['type']]
                        icon_type = type_icon[row['type']]
                        icon = folium.Icon(color=icon_color, icon=icon_type)

                        # Use different icons for hotels, restaurants, and tourist attractions
                        folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)

                        st.write(f"## {index + 1}. {row['displayName.text']}")
                        folium_static(mymap)
                        st.write(f"Rating: {row['rating']}")
                        st.write(f"Address: {row['formattedAddress']}")
                        st.write(f"Website: {row['websiteUri']}")
                        st.write(f"More information: {row['googleMapsUri']}\n")


        def chatbot():
            class Message(BaseModel):
                actor: str
                payload : str

            llm = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo', temperature=0) 

            USER = "user"
            ASSISTANT = "ai"
            MESSAGES = "messages"

                # def initialize_session_state():
            if MESSAGES not in st.session_state:
                st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]
            
            msg: Message
            for msg in st.session_state[MESSAGES]:
                st.chat_message(msg.actor).write(msg.payload)

            # Prompt
            query: str = st.chat_input("Enter a prompt here")

            # Combine info
            df_place['combined_info'] = df_place.apply(lambda row: f"Type: {row['type']}, Name: {row['displayName.text']}. Rating: {row['rating']}. Address: {row['formattedAddress']}. Website: {row['websiteUri']}", axis=1)
            # Load Processed Dataset
            loader = DataFrameLoader(df_place, page_content_column="combined_info")
            docs  = loader.load()

            # Document splitting
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(docs)

            # embeddings model
            # Define the path to the pre-trained model you want to use
            modelPath = "sentence-transformers/all-MiniLM-l6-v2"

            # Create a dictionary with model configuration options, specifying to use the CPU for computations
            model_kwargs = {'device':'cpu'}

            # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
            encode_kwargs = {'normalize_embeddings': False}

            # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
            embeddings = HuggingFaceEmbeddings(
                model_name=modelPath,     # Provide the pre-trained model's path
                model_kwargs=model_kwargs, # Pass the model configuration options
                encode_kwargs=encode_kwargs # Pass the encoding options
            )

            # Vector DB
            vectorstore  = FAISS.from_documents(texts, embeddings)

            template = """ 
            Your job is to assist users in locating a location. 
            From the following context and chat history, assist customers in finding what they are looking for based on their input. 
            Provide three recommendations, along with the address, phone number, website.
            Sort recommendations based on rating and number of user ratings. 
            
            {context}

            chat history: {history}

            input: {question} 
            Your Response:
            """

            prompt = PromptTemplate(
                input_variables=["context","history","question"],
                template=template,
            )

            memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(),
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": prompt,
                    "memory": memory}
        )

            if query:
                st.session_state[MESSAGES].append(Message(actor=USER, payload=str(query)))
                st.chat_message(USER).write(query)

                with st.spinner("Please wait..."):
                    response: str = qa.run(query = query)
                    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
                    st.chat_message(ASSISTANT).write(response)
               # st.write("Chatbot")

        method = st.sidebar.radio(" ",["Search 🔎","ChatBot 🤖","Database 📑"], key="method_app")
        if method == "Search 🔎":
            maps()
        elif method == "ChatBot 🤖":
            chatbot()
        else:
            database()

    st.sidebar.markdown(''' 
        ## Created by: 
        Ahmad Luay Adnani - [GitHub](https://github.com/ahmadluay9) 
        ''')

        
if __name__ == '__main__':
    main()
    





