import openai
from utils.gpt.chat import no_func_chat
from utils.load_data import lim, train_msg

def generate_reason(args, desc_sys, desc_user_0, desc_assis, desc_user_1, desc_ans, reason_user_0, env_id):
    msg = [{"role": "system", "content": desc_sys}, {"role": "user", "content": desc_user_0}, {"role": "assistant", "content": desc_assis}, {"role": "user", "content":desc_user_1}, {"role": "assistant", "content": desc_ans}, {"role": "user", "content": reason_user_0}]
    return no_func_chat(args.gpt[env_id], msg, args.temp[env_id], lim["reason"])