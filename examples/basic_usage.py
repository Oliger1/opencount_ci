# examples/basic_usage.py

from opencount_ci import count_objects, analyze_image

# Use forward slashes OR raw string to avoid escape issues
image_path = r"C:/Users/user/Desktop/apple.jpeg"

print("Count:", count_objects(image_path, mode="auto"))

res = analyze_image(
    image_path,
    mode="auto",
    iterations=20,
    confidence_level=0.9,
    verbose=True
)

print(res)
