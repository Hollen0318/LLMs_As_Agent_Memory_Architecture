import numpy as np

# Get the observation map for all environments, with 3-dimension (object, color, status) and height, width.
def get_world_maps(id, env_sizes):
    # parse the data and create world maps
    world_map = {}
    for env in env_sizes.strip().split("\n"):
        env_id, h, w = map(int, env.strip().split(", "))
        if env_id == id:
            world_map = np.empty((3, h, w), dtype = object)    
            # the three dimensions will be object, color and status, we intiialize them seperately now 
            world_map[0] = np.full((h, w), "-", dtype = object)
            world_map[1] = np.full((h, w), "-", dtype = object)
            world_map[2] = np.full((h, w), "-", dtype = object)
            break
    return world_map

def get_rec(id, env_sizes):

    rec = {}

    for env in env_sizes.strip().split("\n"):
        env_id, h, w = map(int, env.strip().split(','))
        if env_id == id:
            env_view = np.zeros((h, w))
            env_step = np.zeros((h, w))
            env_memo = np.zeros((h, w))
            obj_intr = {}
            obj_intr["toggle"] = np.array([0 for i in range(11)])
            obj_intr["drop off"] = np.array([0 for i in range(11)])
            obj_intr["pick up"] = np.array([0 for i in range(11)])
            obj_view = np.array([0 for i in range(11)])
            break
    rec["env_view"], rec["env_step"], rec["env_memo"], rec["obj_intr"], rec["obj_view"] = env_view, env_step, env_memo, obj_intr, obj_view

    return rec