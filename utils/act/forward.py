def get_front_obj(world_map, pos_x, pos_y, arrow):
    if arrow == "Up/North":
        return int(world_map[0][pos_x - 1][pos_y])
    elif arrow == "Down/South":
        return int(world_map[0][pos_x + 1][pos_y])
    elif arrow == "Left/West":
        return int(world_map[0][pos_x][pos_y - 1])
    elif arrow == "Right/East":
        return int(world_map[0][pos_x][pos_y + 1])
    
def get_front_col(world_map, pos_x, pos_y, arrow):
    if arrow == "Up/North":
        return int(world_map[1][pos_x - 1][pos_y])
    elif arrow == "Down/South":
        return int(world_map[1][pos_x + 1][pos_y])
    elif arrow == "Left/West":
        return int(world_map[1][pos_x][pos_y - 1])
    elif arrow == "Right/East":
        return int(world_map[1][pos_x][pos_y + 1])
    
def get_front_sta(world_map, pos_x, pos_y, arrow):
    if arrow == "Up/North":
        return int(world_map[2][pos_x - 1][pos_y])
    elif arrow == "Down/South":
        return int(world_map[2][pos_x + 1][pos_y])
    elif arrow == "Left/West":
        return int(world_map[2][pos_x][pos_y - 1])
    elif arrow == "Right/East":
        return int(world_map[2][pos_x][pos_y + 1])
    
# Update the position based on the arrow and previous position    
def update_pos(pos_x, pos_y, arrow):
    n_pos_x, n_pos_y = pos_x, pos_y
    if arrow == "Up/North":
        n_pos_x -= 1
    elif arrow == "Down/South":
        n_pos_x += 1
    elif arrow == "Left/West":
        n_pos_y -= 1
    elif arrow == "Right/East":
        n_pos_y += 1
    return n_pos_x, n_pos_y
