from utils.load_data import *
import numpy as np
def memo_add(args, env_memo_rec, row, col):
    if args.neg_memo:
        env_memo_rec[row][col] += lim["memo"]
    else:
        env_memo_rec[row][col] = lim["memo"]
    return env_memo_rec[row][col]

def memo_minus(args, env_memo_rec):
    if args.neg_memo:
        env_memo_rec -= 1
    else:
        env_memo_rec = np.where(env_memo_rec > 0, env_memo_rec - 1, env_memo_rec)
    return env_memo_rec