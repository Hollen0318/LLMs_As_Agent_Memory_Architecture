import openai
import time
from utils.load_data import *

def choose_act(action):
    # Check if param is a list
    if isinstance(action, list):
        # Check if the list has more than 1 element
        if len(action) > 1:
            return action
        else:
            # If it's a list of length 1, extract the first element and try to convert it to int
            try:
                return int(action[0])
            except (ValueError, TypeError):
                raise ValueError("Single element in the list or the parameter itself could not be converted to an integer.")
    else:
        # If it's not a list, try to convert it to int
        try:
            return int(action)
        except (ValueError, TypeError):
            raise ValueError("The parameter could not be converted to an integer.")

def load_api_key(API_KEY):
    openai.api_key = open(API_KEY).read()

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

def generate_action(args, desc_sys, desc_user_0, desc_assis, desc_user_1, desc_ans, reason_user_0, reason_ans, act_user_0, fuc_msg, fuc_desc, env_id):
    msg = [{"role": "system", "content": desc_sys}, 
           {"role": "user", "content": desc_user_0}, 
           {"role": "assistant", "content": desc_assis}, 
           {"role": "user", "content":desc_user_1}, 
           {"role": "assistant", "content": desc_ans}, 
           {"role": "user", "content": reason_user_0}, 
           {"role":"assistant", "content": reason_ans},
           {"role": "user", "content": act_user_0}]
    
    fuc = [{"name": "choose_act", 
            "description": fuc_msg,
            "parameters": {
                "type": "object",
                "properties": {
                    "action":
                    {
                        "type": "integer",
                        "description": fuc_desc
                    }
                },
                "required":["action"]
            }}]
    
    return func_chat(args.gpt[env_id], msg, fuc, args.temp[env_id], lim["action"], args.delay)

def generate_desc(args, desc_sys, desc_user_0, desc_assis, desc_user_1, env_id):
    msg = [{"role": "system", "content": desc_sys}, {"role": "user", "content": desc_user_0}, {"role": "assistant", "content": desc_assis}, {"role": "user", "content":desc_user_1}]
    return no_func_chat(args.gpt[env_id], msg, args.temp[env_id], lim["desc"])

def generate_reason(args, desc_sys, desc_user_0, desc_assis, desc_user_1, desc_ans, reason_user_0, env_id):
    msg = [{"role": "system", "content": desc_sys}, {"role": "user", "content": desc_user_0}, {"role": "assistant", "content": desc_assis}, {"role": "user", "content":desc_user_1}, {"role": "assistant", "content": desc_ans}, {"role": "user", "content": reason_user_0}]
    return no_func_chat(args.gpt[env_id], msg, args.temp[env_id], lim["reason"])

def generate_n_exp(args, desc_sys, desc_user_0, desc_assis, desc_user_1, desc_ans, reason_user_0, reason_ans, n_exp_user_0, env_id):
    msg = [{"role": "system", "content": desc_sys}, {"role": "user", "content": desc_user_0}, {"role": "assistant", "content": desc_assis}, {"role": "user", "content":desc_user_1}, {"role": "assistant", "content": desc_ans}, {"role": "user", "content": reason_user_0}, {"role": "assistant", "content": reason_ans}, {"role": "user", "content": n_exp_user_0}]
    return no_func_chat(args.gpt[env_id], msg, args.temp[env_id], lim["n_exp"])

def generate_s_exp(args, desc_sys, desc_user_0, desc_assis, desc_user_1, desc_ans, reason_user_0, reason_ans, n_exp_user_0, n_exp, s_exp_user_0, env_id):
    msg = [{"role": "system", "content": desc_sys}, {"role": "user", "content": desc_user_0}, {"role": "assistant", "content": desc_assis}, {"role": "user", "content":desc_user_1}, {"role": "assistant", "content": desc_ans}, {"role": "user", "content": reason_user_0}, {"role": "assistant", "content": reason_ans}, {"role": "user", "content": n_exp_user_0}, {"role": "assistant", "content": n_exp}, {"role": "user", "content": s_exp_user_0}]
    return no_func_chat(args.gpt[env_id], msg, args.temp[env_id], lim["s_exp"])