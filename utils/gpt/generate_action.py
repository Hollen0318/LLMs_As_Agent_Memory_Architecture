from utils.gpt.chat import func_chat
from utils.load_data import lim

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