import warnings

def steps_envs_same_length(args):
    if len(args.steps) != len(args.envs):
        warnings.warn(f"The parameter steps length {len(args.steps)} is not the same as the envs {len(args.envs)}, will use the first steps parameter {args.steps[0]} for all environments!", UserWarning)
        args.steps = [args.steps[0] for i in range(len(args.envs))]
    return args

def gpt_envs_same_length(args):
    if len(args.steps) != len(args.envs):
        warnings.warn(f"The parameter gpt length {len(args.gpt)} is not the same as the envs {len(args.envs)}, will use the first steps {args.gpt[0]} for all environments!", UserWarning)
        args.gpt = [args.gpt[0] for i in range(len(args.envs))]
    if args.cross and len(args.gpt) > 1:
        warnings.warn(f"--cross is set to be ture, but gpt length is {len(args.gpt)}, which is larger than 1, so only one gpt version will be used which is the first gpt parameter {args.gpt[0]}", UserWarning)
        args.gpt = [args.gpt[0] for i in range(len(args.envs))]
    return args

def exp_envs_same_length(args):
    if len(args.exp_src) != len(args.envs):
        warnings.warn(f"The experience source length {len(args.exp_src)} is not the same as the envs {len(args.envs)}, will use the first exp source parameter {args.exp_src[0]} for all environments", UserWarning)
        args.exp_src = [args.exp_src[0] for i in range(len(args.envs))]
    return args

def temp_envs_same_length(args):
    if len(args.temp) != len(args.envs):
        warnings.warn(f"The API temp length {len(args.temp)} is not the same as the envs {len(args.envs)}, will use the first temp parameter {args.temp[0]} for all environments", UserWarning)
        args.temp = [args.temp[0] for i in range(len(args.envs))]
    if args.cross and len(args.temp) > 1:
        warnings.warn(f"--cross is set to be ture, but temp length is {len(args.temp)}, which is larger than 1, so only one temp value will be used which is the first temp parameter {args.temp[0]}", UserWarning)
        args.temp = [args.temp[0] for i in range(len(args.envs))]
    return args

def examine(args):
    args = steps_envs_same_length(args)
    args = gpt_envs_same_length(args)
    args = exp_envs_same_length(args)
    args = temp_envs_same_length(args)
    return args