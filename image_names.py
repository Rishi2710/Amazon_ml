import pandas as pd
from urllib.parse import urlparse
import os

# Load the dataset
train_csv_path = "/Users/rishit/Desktop/student_resource 3/dataset/train.csv"  # Replace with your train.csv path
df = pd.read_csv(train_csv_path)

# Function to extract the image name from the image link
def extract_image_name(url):
    # Parse the URL and get the path
    parsed_url = urlparse(url)
    # Get the image name from the path
    image_name = os.path.basename(parsed_url.path)
    return image_name

# Apply the function to the 'image_link' column and store the results in a new column 'image_name'
df['image_name'] = df['image_link'].apply(extract_image_name)

# Drop the 'image_link' column if you only want the image name
df = df.drop(columns=['image_link'])

# Save the updated DataFrame to a new CSV if needed
df.to_csv('train_with_image_names.csv', index=False)

# Display the updated DataFrame
print(df.head())