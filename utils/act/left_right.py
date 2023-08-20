def left_arrow(direction):
    r_direction = ""
    if direction == "Up/North":
        r_direction = "Left/West"
    elif direction == "Down/South":
        r_direction = "Right/East"
    elif direction == "Left/West":
        r_direction = "Down/South"
    elif direction == "Right/East":
        r_direction = "Up/North"
    if r_direction == "":
        print(f"failed to detect {direction} in left_arrow")
    return r_direction

def right_arrow(direction):
    r_direction = ""
    if direction == "Up/North":
        r_direction = "Right/East"
    elif direction == "Down/South":
        r_direction = "Left/West"
    elif direction == "Left/West":
        r_direction = "Up/North"
    elif direction == "Right/East":
        r_direction = "Down/South"
    if r_direction == "":
        print(f"failed to detect {direction} in right_arrow")
    return r_direction