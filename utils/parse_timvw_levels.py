import os
import os.path
import json
import pandas as pd


def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')


def clean_name(text):
    ''' Clean text to be used as identifier in filename
    '''
    return text.replace("_", "-").replace(" ", "-").lower()


def get_boxes(layout):
    # print(get_boxes(original_json['SokobanLevels']['LevelCollection']['Level'][1]['L']))
    all_boxes = []
    for y, row in enumerate(layout):
        for x, obj in enumerate(row):
            if obj == "$" or obj == '*':  # $ if on floor, * if on goal
                all_boxes.append({'x': x, 'y': y, 'state': obj})
    return (all_boxes)


def get_start_position(layout):
    # print(get_start_position(original_json['SokobanLevels']['LevelCollection']['Level'][0]['L']))
    for y, row in enumerate(layout):
        for x, symbol in enumerate(row):
            if symbol == '@' or symbol == '+':    # @ if on floor, + if on goal
                return {"x": x, "y": y}
    return None


def parse_collection_json(input_folder, out_folder, collection_name, collection_id="NA", keep_collection_together=False):
    # First specify file paths
    input_file = os.path.join(input_folder, collection_name + ".json")
    output_folder = os.path.join(out_folder, collection_name)
    collection_name = clean_name(collection_name)

    # Next, load the original json file which has multiple levels
    with open(input_file, 'r') as f:
        original_json = json.load(f)
    # print(original_json['SokobanLevels']['LevelCollection']['Level'][0])

    # Get ready to parse individual levels
    levels = []
    level_id = 1
    collection_name = original_json['SokobanLevels']['Title']
    for level in original_json['SokobanLevels']['LevelCollection']['Level']:
        boxes = get_boxes(level['L'])
        start_position = get_start_position(level['L'])
        # build desired level spec
        spec = {
            "collection_id": collection_id,
            "collection_name": clean_name(str(collection_name)),
            "level_id": level_id,
            "level_name": clean_name(str(level['Id'])),
            "width": level['Width'],
            "height": level['Height'],
            "layout": level['L'],
            "start_position": start_position,
            "boxes": boxes
        }
        print(f"Exporting {spec['collection_name']}, {spec['level_name']}")
        # Next, export this level as json with systematic file name, to collection subfolder
        with safe_open_w(os.path.join(output_folder, f"collection-{collection_id}_level-{level_id}.json")) as f:
            json.dump(spec, f, indent=2)
        # If save entire collection, store levels in list
        if keep_collection_together:
            levels.append(spec)
        # increment level counter by 1
        level_id += 1

    # If save entire collection, export all levels in single json to main output folder
    if keep_collection_together:
        output_collection_file = os.path.join(
            out_folder, collection_name + ".json")
        with safe_open_w(output_collection_file) as f:
            json.dump(levels, f, indent=2)


def convert_collection_to_df(original_file):
    with open(original_file, 'r') as f:
        # load data from json file as object
        data = json.load(f)
    # extract levels and collection metadata
    levels_data = pd.json_normalize(
        data=data['SokobanLevels']['LevelCollection'],
        record_path='Level',
        record_prefix='Level_')
    levels_data['Collection_Description'] = data['SokobanLevels']['Description']
    levels_data['Collection_Name'] = clean_name(
        str(data['SokobanLevels']['Title']))
    # Refine ID columns
    levels_data.rename(columns={'Level_L': 'Level_Layout',
                                'Level_Id': 'Level_Name'}, inplace=True)
    # Force level names to be strings & in same format as file names
    levels_data['Level_Name'] = levels_data['Level_Name'].astype(
        'string').str.lower()
    levels_data['Level_Name'] = levels_data['Level_Name'].str.replace(
        "_", "-").replace(" ", "-")
    # print(levels_data.head())

    # sort columns by name
    levels_data.sort_index(axis=1, inplace=True)

    return levels_data


if __name__ == '__main__':

    ORIGINAL_PATH = '/Users/junyichu/Documents/projects/timvw sokoban master js-icteam_levels'
    OUTPUT_PATH = 'stimuli/parsed_stim'

    # ---- Aggregate all collections to a single pandas dataframe
    original_files = [f for f in os.listdir(
        ORIGINAL_PATH) if f.endswith('.json')]
    filepaths = [os.path.join(ORIGINAL_PATH, f) for f in original_files]
    # parse all collections & bind dataframes
    df = pd.concat(map(convert_collection_to_df, filepaths))
    # df.info()
    df.to_csv(os.path.join("stimuli/all-levels.tsv"),
              sep='\t', index=False)

    # ---- to export collection to stimuli files
    collection_id = 1
    for f in original_files:
        COLLECTION_NAME = f.replace('.json', '')
        print(f"Processing {COLLECTION_NAME}")
        parse_collection_json(ORIGINAL_PATH, OUTPUT_PATH,
                              COLLECTION_NAME, collection_id, keep_collection_together=False)
        collection_id += 1
