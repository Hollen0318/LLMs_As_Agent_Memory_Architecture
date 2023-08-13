import warnings

def steps_envs_same_length(args):
    if len(args.steps) != len(args.envs):
        warnings.warn(f"The parameter steps length {len(args.steps)} is not the same as the envs {len(args.envs)}, will use default steps {args.steps[0]} for all environments!", UserWarning)
    args.steps = [args.steps[0] for i in range(len(args.envs))]
def examine_args(args):
    steps_envs_same_length(args)