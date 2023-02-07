import openai
import pandas as pd
import numpy as np
from getpass import getpass
from openai.embdeddings_utils import get_embedding
from openai.embedding_utils import cosine_similarity


def get_api_key():
    open.api_key = getpass("Enter your OpenAI API key: ")

def get_data():
    df = pd.read_csv('data.csv')
    df['text'] = df['text'].apply(lambda x: x.lower())
    return df

def get_embeddings(df):
    df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    df.to_csv('word_embeddings.csv')

earnings_df = get_data()
df = get_embeddings(raw_data)

search_term = input("Enter a search term: ")
search_term_vector = get_embedding(search_term, engine='text-embedding-ada-002')

df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
