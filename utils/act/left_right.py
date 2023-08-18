def left_arrow(direction):
    r_direction = ""
    if direction == "Up/North":
        return_direction = "Left/West"
    elif direction == "Down/South":
        r_direction = "Right/East"
    elif direction == "Left/West":
        r_direction = "Down/South"
    elif direction == "Right/East":
        r_direction = "Up/North"
    return r_direction

def right_arrow(direction):
    r_direction = ""
    if direction == "Up/North":
        return_direction = "Right/East"
    elif direction == "Down/South":
        r_direction = "Left/West"
    elif direction == "Left/West":
        r_direction = "Up/North"
    elif direction == "Right/East":
        r_direction = "Down/South"
    return r_direction