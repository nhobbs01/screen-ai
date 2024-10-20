import pandas as pd
import json
from PIL import Image, ImageDraw
import re

base_image_path = 'C:/Users/milia.io/Downloads/combined/'
base_csv_path = '../screen_annotation/'

def annotate_screen(id, image_schema):
    with Image.open(base_image_path+f"{id}.jpg") as im:
        (labels, coords) = parse_image_schema(image_schema)
        draw_bounding_boxes_on_image(im, labels, coords)
        return im

def draw_bounding_boxes_on_image(im, labels, coords):
    for label, coord in zip(labels, coords):
        (x_min, x_max, y_min, y_max) = coord
        # Normalized bounding box coordinates (0 to 999)
        original_width = im.width
        original_height = im.height
        # Convert normalized coordinates to original image size
        left = int(x_min) * (original_width / 1000)
        top = int(y_min) * (original_height / 1000)
        right = int(x_max) * (original_width / 1000)
        bottom = int(y_max) * (original_height / 1000)
        draw = ImageDraw.Draw(im)
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        draw.text((left+5, top-10), label, fill='red')

def parse_image_schema(schema):
    # Convert to uppercase
    pattern = r'(^|[,\(]\s*)([A-Z]+)'
    text_matches = re.findall(pattern, schema)

    # Extract and print the all-caps text
    all_caps_texts = [match[1].strip() for match in text_matches if match[1].strip()]
        
    pattern = r'(\d+\s+\d+\s+\d+\s+\d+)(?=,|\))'
    coord_matches = re.findall(pattern, schema)
    coords = [tuple(match.split(' ')) for match in coord_matches]
    
    # Output the results
    return (all_caps_texts, coords)

def get_annotated_image_by_number(num, split):
    if(split == 'train'):
        df = pd.read_csv(base_csv_path+'train.csv')
    else:
        df = pd.read_csv(base_csv_path+'test.csv')
    
    return annotate_screen(df.iloc[num]['screen_id'], df.iloc[num]['screen_annotation'])

if __name__ == "__main__":
    im = get_annotated_image_by_number(1, 'train')
    im.show()
