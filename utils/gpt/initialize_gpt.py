import openai

def load_api_key(API_KEY):
    openai.api_key = open(API_KEY).read()