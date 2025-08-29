import os
import pandas as pd
import json
import warnings as wn

metadata_path = '/home/notebook/data/group/HUIMING/easyanimate_training/human_free_90K.csv'

if __name__ == '__main__':
    metadata = pd.read_csv(metadata_path, sep=',').to_dict(orient='records')

    print(f">>>> Totally {len(metadata)} rows to check and convert")

    fileLostCounter = 0
    new_metadata = []
    
    for i, video_info in enumerate(metadata, start=1):
        path = video_info.get("file_path", None)
        text = video_info.get("text", None)
        if path is None or text is None:
            raise ValueError(f">>>> Error: Key Value of row {i} is empty, video_info = {video_info}")
        if os.path.exists(path):
            file = {}
            file["file_path"] = path
            file["text"] = text
            for key, value in video_info.items():
                if key in ["file_path", "text"]:
                    continue
                file[key] = value
            new_metadata.append(file)
        else:
            # raise FileNotFoundError(f">>>> Error: Row {i} the file not exists: video_info = {video_info}")
            fileLostCounter += 1
        if i % 10000 == 0:
            print(f">>>> {i} rows has done, currently lost {fileLostCounter} files.")
    
    with open("./metadata.json", "w", encoding="utf-8") as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=2)
    
    print(f">>>> File Lost: {fileLostCounter}")
    print(f">>>> Valid files: {len(new_metadata)}")
    print(">>>> ----- CHECKING AND CONVERTING FINISHED ----- <<<<")
