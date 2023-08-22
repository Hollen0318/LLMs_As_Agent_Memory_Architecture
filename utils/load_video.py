import json

# Load the utilities JSON
def load_json(address):
    with open(address, 'r', encoding='utf-8') as f:
        return json.load(f)

font_size = load_json(r"data/input/video/font_size.json")
content = load_json(r"data/input/video/content.json")
pos = load_json(r"data/input/video/pos.json")
img_size = load_json(r"data/input/video/img_size.json")