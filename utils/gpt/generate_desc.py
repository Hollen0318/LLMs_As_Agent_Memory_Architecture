from utils.gpt.chat import no_func_chat
from utils.load_data import lim

def generate_desc(args, desc_sys, desc_user_0, desc_assis, desc_user_1, env_id):
    msg = [{"role": "system", "content": desc_sys}, {"role": "user", "content": desc_user_0}, {"role": "assistant", "content": desc_assis}, {"role": "user", "content":desc_user_1}]
    return no_func_chat(args.gpt[env_id], msg, args.temp[env_id], lim["desc"])