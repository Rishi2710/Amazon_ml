import os
import cv2
import numpy as np

def preprocess_image_final(image_path, output_path):
    # Step 1: Load the image
    image = cv2.imread(image_path)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply CLAHE with lower clip limit (to avoid aggressive contrast)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(gray)

    # Step 4: Apply mild denoising (Non-Local Means Denoising)
    denoised = cv2.fastNlMeansDenoising(enhanced_contrast, h=5, templateWindowSize=7, searchWindowSize=21)

    # Step 5: Apply very light sharpening
    sharpen_kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # Step 6: Optionally upscale the image to improve OCR for smaller fonts
    upscale = cv2.resize(sharpened, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Step 7: Save the final preprocessed image
    cv2.imwrite(output_path, upscale)

def preprocess_folder(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Preprocess and save the image
            preprocess_image_final(input_path, output_path)
            print(f"Preprocessed image saved to {output_path}")

# Example usage
input_folder = "/Users/rishit/Desktop/student_resource 3/images"  # Replace with your image folder path
output_folder = "/Users/rishit/Desktop/student_resource 3/pre_all"  # Replace with your output folder path

# Preprocess all images in the input folder
preprocess_folder(input_folder, output_folder)