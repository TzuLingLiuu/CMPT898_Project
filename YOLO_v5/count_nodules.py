import os
import pandas as pd # type: ignore



def count_bounding_boxes_with_ground_truth(input_folder, input_folder_2, metadata_csv, output_csv):
   
    metadata = pd.read_csv(metadata_csv)
    

    ground_truth_mapping = dict(zip(metadata["Image ID1"].astype(str), metadata["Number of nodules (count)"]))
    

    results = []
    
    
    input_folder_2_counts = {}
    for filename in os.listdir(input_folder_2):
        if filename.endswith(".txt"):
            base_name = ''.join(filter(str.isdigit, filename))
            file_path = os.path.join(input_folder_2, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                input_folder_2_counts[base_name] = len(lines)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            base_name = ''.join(filter(str.isdigit, filename))
          
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                num_bounding_boxes = len(lines)
            
            ground_truth = ground_truth_mapping.get(base_name, None)
            
         
            manual_annotated_ground_truth = input_folder_2_counts.get(base_name, None)
            
            
            results.append({
                "Image_ID": base_name,
                "Ground_Truth_Count": int(ground_truth) if ground_truth is not None else None,
                "Manual_Annotated_Ground_Truth": manual_annotated_ground_truth,
                "Predict_Count": num_bounding_boxes
                
            })
    
   
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Result is in {output_csv}")

input_folder = "/student/ywa826/project/CMPT898/Project/YOLO_v5/test_outputs/labels"
input_folder_2 = "/student/ywa826/project/CMPT898/Project/YOLO_v5/YOLODataset_Test/labels/test"
metadata_csv = "/student/ywa826/project/CMPT898/Project/YOLO_v5/Metadata_by_Image_ID.csv"
output_csv = "/student/ywa826/project/CMPT898/Project/YOLO_v5/test_outputs/nodule_count.csv"

count_bounding_boxes_with_ground_truth(input_folder, input_folder_2, metadata_csv, output_csv)


