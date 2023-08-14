from utils.load_data import *
import numpy as np
def memo_add(args, env_memo_rec, env_id, row, col):
    if args.neg_memo:
        env_memo_rec[env_id][row][col] += lim["memo"]
    else:
        env_memo_rec[env_id][row][col] = lim["memo"]
    return env_memo_rec[env_id][row][col]

def memo_minus(args, env_memo_rec, env_id):
    if args.neg_memo:
        env_memo_rec[env_id] -= 1
    else:
        env_memo_rec[env_id] = np.where(env_memo_rec[env_id] > 0, env_memo_rec[env_id] - 1, env_memo_rec[env_id])
    return env_memo_rec[env_id]