import warnings

def steps_envs_same_length(args):
    if len(args.steps) != len(args.envs):
        warnings.warn(f"The parameter steps length {len(args.steps)} is not the same as the envs {len(args.envs)}, will use default steps {args.steps[0]} for all environments!", UserWarning)
        args.steps = [args.steps[0] for i in range(len(args.envs))]
    return args

def gpt_envs_same_length(args):
    if len(args.steps) != len(args.envs):
        warnings.warn(f"The parameter gpt length {len(args.gpt)} is not the same as the envs {len(args.envs)}, will use default steps {args.gpt[0]} for all environments!", UserWarning)
        args.gpt = [args.gpt[0] for i in range(len(args.envs))]
    if args.cross and len(args.gpt) > 1:
        warnings.warn(f"--cross is set to be ture, but gpt length is {len(args.gpt)}, which does not equal to envs length {len(args.envs)}, so only one gpt version will be used which is {args.gpt[0]}")
        args.gpt = [args.gpt[0] for i in range(len(args.envs))]
    return args

def examine(args):
    args = steps_envs_same_length(args)
    args = gpt_envs_same_length(args)
    return args