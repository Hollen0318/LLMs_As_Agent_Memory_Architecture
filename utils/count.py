import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def tokens_count(text):
    return len(encoding.encode(text))