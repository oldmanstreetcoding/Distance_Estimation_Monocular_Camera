import pandas as pd
import numpy as np

# Load the source data from the text file
def load_source_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(" ")
            image_name = parts[0]
            objects = parts[1:]
            for obj in objects:
                obj_parts = obj.split(",")
                data.append({
                    "image": image_name,
                    "xmin": int(obj_parts[0]),
                    "ymin": int(obj_parts[1]),
                    "xmax": int(obj_parts[2]),
                    "ymax": int(obj_parts[3]),
                    "cls": int(obj_parts[4]),
                    "dist": float(obj_parts[5])
                })
    return pd.DataFrame(data)

# Load the target data from the CSV file
def load_target_data(file_path):
    return pd.read_csv(file_path)

# Find the closest match for bounding boxes, accounting for rounding differences
def find_closest_match(row, source_df, tolerance=1.0):
    # Extract bounding box from the target row
    target_bbox = {
        "xmin": row["xmin"],
        "ymin": row["ymin"],
        "xmax": row["xmax"],
        "ymax": row["ymax"]
    }
    # Filter source data for the same image and class
    filtered_source = source_df[
        (source_df["image"] == row["filename"].replace(".txt", ".png")) & 
        (source_df["cls"].map(class_mapping) == row["class"])
    ]
    if filtered_source.empty:
        return None

    # Calculate Euclidean distance between bounding boxes
    filtered_source["bbox_distance"] = filtered_source.apply(
        lambda src: np.sqrt(
            (src["xmin"] - target_bbox["xmin"]) ** 2 +
            (src["ymin"] - target_bbox["ymin"]) ** 2 +
            (src["xmax"] - target_bbox["xmax"]) ** 2 +
            (src["ymax"] - target_bbox["ymax"]) ** 2
        ),
        axis=1
    )
    # Find the closest match within the tolerance
    closest_match = filtered_source[filtered_source["bbox_distance"] <= tolerance]
    if not closest_match.empty:
        return closest_match.iloc[0]["dist"]
    return None

# Update target data by adding a new 'dist' column
def update_target_with_dist(target_df, source_df):
    target_df["dist"] = None  # Add a new column for dist
    updated_rows = []
    for _, row in target_df.iterrows():
        dist_value = find_closest_match(row, source_df)
        row["dist"] = dist_value if dist_value is not None else None  # Keep existing data intact
        updated_rows.append(row)
    return pd.DataFrame(updated_rows)

# Save the updated target data to a CSV file
def save_updated_target(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)

# Main execution
source_file = "dist_yolo_kitty_train_distance_groundtruth.txt"  # Source text file
target_file = "annotations.csv"  # Target CSV file
output_file = "updated_target_with_dist.csv"  # Output file

# Load data
source_df = load_source_data(source_file)
target_df = load_target_data(target_file)

# Create a mapping of class IDs to class names
class_mapping = {
    0: "Pedestrian",
    1: "Car",
    2: "Van",
    3: "Truck",
    4: "Person_sitting",
    5: "Cyclist",
    6: "Tram"
}

# Update target data with a new 'dist' column
updated_target_df = update_target_with_dist(target_df, source_df)

# Save the updated target file
save_updated_target(updated_target_df, output_file)
print(f"Updated target file saved as '{output_file}'")
