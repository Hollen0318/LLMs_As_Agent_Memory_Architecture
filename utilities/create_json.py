import os
import json

def read_files_from_dir(directory):
    data_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as f:
                data_dict[filename[:-4]] = f.read() # remove '.txt' from filename for key
    return data_dict

def save_dict_to_json(dictionary, output_file):
    with open(output_file, 'w') as f:
        json.dump(dictionary, f, indent=4)

def load_dict_from_json(input_file):
    with open(input_file, 'r') as f:
        return json.load(f)

# directory = './' # replace with your directory
# output_file = 'merged_data.json' # replace with your desired output file name
input_file = 'merged_data.json' # replace with your file name

# read txt files and save to dictionary
# data_dict = read_files_from_dir(directory)

# save dictionary to json
# save_dict_to_json(data_dict, output_file)

# # load dictionary from json
loaded_dict = load_dict_from_json(input_file)

# # example of usage with .format()
# past_actions = "went to the gym, studied, cooked dinner" # replace with your actual data
# msg = loaded_dict['example_key'] # replace 'example_key' with your actual key
# filled_text = msg.format(past_actions)
# print(filled_text)
s = loaded_dict['desc_msg']
m = s.format("haha", "haha", "haha", "haha", "haha", "haha", "haha", "haha")
print(m)