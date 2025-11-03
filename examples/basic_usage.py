from opencount_ci import count_objects, analyze_image

print("Count:", count_objects("C:\Users\user\Desktop\apple.jpeg", mode="auto"))
res = analyze_image("C:\Users\user\Desktop\apple.jpeg", mode="auto", iterations=20, confidence_level=0.9, verbose=True)
print(res)
