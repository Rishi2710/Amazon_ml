import os
import pandas as pd
from paddleocr import PaddleOCR

# Step 1: Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', cpu_threads=4)  # Adjust threads based on your system

# Step 2: Function to extract text from an image using PaddleOCR
def extract_text_from_image(image_path):
    result = ocr.ocr(image_path, cls=True)
    extracted_text = " ".join([line[1][0] for line in result[0]]) if result else ""
    return extracted_text

# Step 3: Function to loop through folder and process images
def process_images_and_update_csv(image_folder, csv_path, output_csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Ensure the CSV has a column for image names and create a new column for extracted text
    if 'image_name' not in df.columns:
        raise ValueError("CSV does not contain 'image_name' column.")
    
    # Loop through each image in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)

            # Extract text from the image
            extracted_text = extract_text_from_image(image_path)
            print(f"Extracted text for {filename}: {extracted_text}")

            # Match the image name in the DataFrame and update the corresponding row
            image_name_in_csv = filename  # Assuming image_name column matches filenames
            df.loc[df['image_name'] == image_name_in_csv, 'extracted_text'] = extracted_text

    # Save the updated DataFrame to a new CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

# Step 4: Main function to run the code
if __name__ == "__main__":
    image_folder = "/Users/rishit/Desktop/student_resource 3/pre_all"  # Folder with pre-processed images
    csv_path = "/Users/rishit/Desktop/student_resource 3/train_with_image_names.csv"  # Path to your CSV file

    # Process images and update CSV
    process_images_and_update_csv(image_folder, csv_path, csv_path)  # Same file for input and output