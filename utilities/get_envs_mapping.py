import os
import re

def find_environment_names(directory, output_file):
    env_pattern = re.compile(r'MiniGrid-\w+(-\d+x\d+|-v\d|-[A-Za-z]+)*\d')
    env_names = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    contents = f.read()
                    matches = env_pattern.findall(contents)
                    for fullmatch in re.finditer(env_pattern, contents):
                        env_names.append(fullmatch.group())
    print(env_names)
    # Write the environment names to the output file
    index = 0
    if env_names:  # Only write to file if env_names is not empty
        with open(output_file, 'w') as f:
            for name in env_names:
                f.write (str(index) + ', ')
                f.write(name + '\n')
                index += 1
# Example usage:
find_environment_names(r'..\Minigrid\minigrid\envs', 'envs_mapping.txt')