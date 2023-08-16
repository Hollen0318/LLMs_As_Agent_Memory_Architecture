import numpy as np
from utils.load_data import *
from utils.memo import memo_add, memo_minus
# 1. We need to update the world map in object level, which we should initialize using pos_x, pos_y, arrow, obs, world_map, env_memo_rec

def update_world_map(args, world_map, pos_x, pos_y, arrow, obs, rec):
    # With the new obs, we should first update the env_memo_rec, as it will determine which parts of world map will show
    p_obj, p_col, p_sta = world_map[0][pos_x][pos_y], world_map[1][pos_x][pos_y], world_map[2][pos_x][pos_y]
    image = obs['image'].transpose(1,0,2)
    env_step_rec, env_memo_rec, env_view_rec, obj_view_rec = rec["env_step"], rec["env_memo"], rec["env_view"], rec["obj_view"]
    # 0. Deduct the env_memo_rec by 1 unless 0
    env_memo_rec  = memo_minus(args, env_memo_rec)
    if arrow == "Right/East":
        # It means the agent is facing right, so we update the env_memo_rec accordingly, specifically we 
        # set args.memo to the corresponding observed area while all other values are deducted by 1 unless equals to 0
        # In addition we increment one to the env_view_rec which is basically recording how many times agent has seen that area
        rotated_image_obj = np.rot90(image[:, :, 0], k = -1)
        rotated_image_col = np.rot90(image[:, :, 1], k = -1)
        rotated_image_sta = np.rot90(image[:, :, 2], k = -1)
        # 1. Update the env_view_rec, env_memo_rec, obj_view_rec, world_map in three channels
        for row in range(max(0, pos_x - args.view // 2), min(env_view_rec.shape[0], pos_x + args.view // 2 + 1)):
            for col in range(pos_y, min(env_view_rec.shape[1], pos_y + args.view)):
                # If the object is not unseen, we record it to the env_memo, env_view, obj_view, world_map
                obj_name = rotated_image_obj[row - (pos_x - args.view // 2)][col - pos_y]
                col_name = rotated_image_col[row - (pos_x - args.view // 2)][col - pos_y]
                sta_name = rotated_image_sta[row - (pos_x - args.view // 2)][col - pos_y]
                if obj_name != 0:
                    env_memo_rec[row][col] = memo_add(args, env_memo_rec, row, col)
                    env_view_rec[row][col] += 1
                    # Update the world_map object level to be what's seen, except in agent's position we set it to be the arrow
                    world_map[0][row][col] = str(obj_name)
                    world_map[1][row][col] = str(col_name)
                    world_map[2][row][col] = str(sta_name)
                obj_view_rec[obj_name] += 1
    elif arrow == "Up/North":
        # It means the agent is facing north, so we update the env_memo_rec accordingly, specifically we 
        # set args.memo to the corresponding observed area while all other values are deducted by 1 unless equals to 0
        # In addition we increment one to the env_view_rec which is basically recording how many times agent has seen that area
        rotated_image_obj = image[:, :, 0]
        rotated_image_col = image[:, :, 1]
        rotated_image_sta = image[:, :, 2]
        # 1. Update the env_view_rec, env_memo_rec, obj_view_rec, world_map in three channels
        for row in range(max(0, pos_x - args.view + 1), pos_x + 1):
            for col in range(max(0, pos_y - args.view // 2), min(env_view_rec.shape[1], pos_y + args.view // 2 + 1)):
                # If the object is not unseen, we record it to the env_memo, env_view, obj_view, world_map
                obj_name = rotated_image_obj[row - (pos_x - args.view + 1)][col - (pos_y - args.view // 2)]
                col_name = rotated_image_col[row - (pos_x - args.view + 1)][col - (pos_y - args.view // 2)]
                sta_name = rotated_image_sta[row - (pos_x - args.view + 1)][col - (pos_y - args.view // 2)]
                if obj_name != 0:
                    env_memo_rec[row][col] = memo_add(args, env_memo_rec, row, col)
                    env_view_rec[row][col] += 1
                    # Update the world_map object level to be what's seen, except in agent's position we set it to be the arrow
                    world_map[0][row][col] = str(obj_name)
                    world_map[1][row][col] = str(col_name)
                    world_map[2][row][col] = str(sta_name)
                obj_view_rec[obj_name] += 1
    elif arrow == "Down/South":
        # It means the agent is facing south, so we update the env_memo_rec accordingly, specifically we 
        # set args.memo to the corresponding observed area while all other values are deducted by 1 unless equals to 0
        # In addition we increment one to the env_view_rec which is basically recording how many times agent has seen that area
        rotated_image_obj = np.rot90(image[:, :, 0], k = -2)
        rotated_image_col = np.rot90(image[:, :, 1], k = -2)
        rotated_image_sta = np.rot90(image[:, :, 2], k = -2)
        # 1. Update the env_view_rec, env_memo_rec, obj_view_rec, world_map in three channels
        for row in range(pos_x, min(env_view_rec.shape[0], pos_x + args.view)):
            for col in range(max(0, pos_y - args.view // 2), min(env_view_rec.shape[1], pos_y + args.view // 2 + 1)):
                # If the object is not unseen, we record it to the env_memo, env_view, obj_view, world_map
                obj_name = rotated_image_obj[row - pos_x][col - (pos_y - args.view // 2)]
                col_name = rotated_image_col[row - pos_x][col - (pos_y - args.view // 2)]
                sta_name = rotated_image_sta[row - pos_x][col - (pos_y - args.view // 2)]
                if obj_name != 0:
                    env_memo_rec[row][col] = memo_add(args, env_memo_rec, row, col)
                    env_view_rec[row][col] += 1
                    # Update the world_map object level to be what's seen, except in agent's position we set it to be the arrow
                    world_map[0][row][col] = str(obj_name)
                    world_map[1][row][col] = str(col_name)
                    world_map[2][row][col] = str(sta_name)
                obj_view_rec[obj_name] += 1
    elif arrow == "Left/West":
        # It means the agent is facing right, so we update the env_memo_rec accordingly, specifically we 
        # set args.memo to the corresponding observed area while all other values are deducted by 1 unless equals to 0
        # In addition we increment one to the env_view_rec which is basically recording how many times agent has seen that area
        rotated_image_obj = np.rot90(image[:, :, 0], k = 1)
        rotated_image_col = np.rot90(image[:, :, 1], k = 1)
        rotated_image_sta = np.rot90(image[:, :, 2], k = 1)
        # 1. Update the env_view_rec, env_memo_rec, obj_view_rec, world_map in three channels
        for row in range(max(0, pos_x - args.view // 2), min(env_view_rec.shape[0], pos_x + args.view // 2 + 1)):
            for col in range(max(0, pos_y - args.view), pos_y + 1):
                # If the object is not unseen, we record it to the env_memo, env_view, obj_view, world_map
                obj_name = rotated_image_obj[row - (pos_x - args.view // 2)][col - (pos_y - args.view + 1)]
                col_name = rotated_image_col[row - (pos_x - args.view // 2)][col - (pos_y - args.view + 1)]
                sta_name = rotated_image_sta[row - (pos_x - args.view // 2)][col - (pos_y - args.view + 1)]
                if obj_name != 0:
                    env_memo_rec[row][col] = memo_add(args, env_memo_rec, row, col)
                    env_view_rec[row][col] += 1
                    # Update the world_map object level to be what's seen, except in agent's position we set it to be the arrow
                    world_map[0][row][col] = str(obj_name)
                    world_map[1][row][col] = str(col_name)
                    world_map[2][row][col] = str(sta_name)
                obj_view_rec[obj_name] += 1
    # We update all the positions to -, and return the last position so can be used in the experience comparison
    world_map[0][pos_x][pos_y] = "Yourself"
    world_map[1][pos_x][pos_y] = "Yourself"
    world_map[2][pos_x][pos_y] = "Yourself"
    env_step_rec[pos_x][pos_y] += 1
    for row in range(env_memo_rec.shape[0]):
        for col in range(env_memo_rec.shape[1]):
            if env_memo_rec[row][col] <= 0:
                world_map[0][row][col] = "-"
                world_map[1][row][col] = "-"
                world_map[2][row][col] = "-"
    # Remember not to update previous obj, col, sta when the agent has not make a move

    c_world_map = compose_world_map(world_map)
    
    return p_obj, p_col, p_sta, c_world_map

def compose_world_map(world_map):
    c_world_map = world_map.copy()
    height = world_map.shape[0]
    width = world_map.shape[1]
    for i in range(height):
        for j in range(width):
            c_world_map[i][j] = f"{obs_rep['color'][world_map[1][i][j]]}{obs_rep['object'][world_map[0][i][j]]}{obs_rep['status'][world_map[2][i][j]]}"
    return c_world_map