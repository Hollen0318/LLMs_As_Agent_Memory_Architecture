import openai
import time
from utils.load_data import gpt_map

def no_func_chat(model, msg, temp, lim, delay):
    while True:
        try:
            rsp = openai.ChatCompletion.create(
                model = gpt_map[model],
                messages = msg,
                temperature = temp,
                max_tokens = lim
            )
            ans = rsp["choices"][0]["messages"]["content"]
            break
        except Exception as e:
            time.sleep(delay)
            delay *= 2

    return ans

def func_chat(model, msg, fuc, temp, lim, delay):
    while True:
        try:
            rsp = openai.ChatCompletion.create(
                model = gpt_map[model],
                messages = msg,
                functions = fuc,
                function_call = "auto",
                temperature = temp,
                max_tokens = lim
            )
            ans = rsp["choices"][0]["messages"]["content"]
            break
        except Exception as e:
            time.sleep(delay)
            delay *= 2

    return ans