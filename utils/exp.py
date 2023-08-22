# Based on the training or evaluation, we initialize the exp,
# If it's evaluation we return a list of experience that is 
# optimal to the evaluation settings
# If it is training, based on cross, we initalize one exp or use a list of exp
# When we do the update, we will determine which exp to use again
def initialize_exp(args):
    if args.eval:
        return get_eval_exp(args)
    else:
        if args.cross:
            if args.retrain:
                exp = open(args.exp_src[0]).read()
                return exp
            else:
                exp = ""
        else:
            if args.retrain:
                exp = ""
            else:
                exp = ["" for i in range(len(args.envs))]
    return exp

def train_exp(args, env_id, exp):
    if args.cross:
        return exp
    else:
        return exp[env_id]

def get_eval_exp(args):
    if args.exp_src is not None:
        exp = open(args.exp_src[0]).read()
        return exp
    else:
        exp = ""
    return exp