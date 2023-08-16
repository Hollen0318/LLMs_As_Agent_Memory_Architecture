def choose_act(action):
    # Check if param is a list
    if isinstance(action, list):
        # Check if the list has more than 1 element
        if len(action) > 1:
            return action
        else:
            # If it's a list of length 1, extract the first element and try to convert it to int
            try:
                return int(action[0])
            except (ValueError, TypeError):
                raise ValueError("Single element in the list or the parameter itself could not be converted to an integer.")
    else:
        # If it's not a list, try to convert it to int
        try:
            return int(action)
        except (ValueError, TypeError):
            raise ValueError("The parameter could not be converted to an integer.")