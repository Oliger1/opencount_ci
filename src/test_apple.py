from opencount_ci import count_objects, analyze_image, configure_detection
from collections import Counter

# Test 1: Numërim i thjeshtë
print("Test 1: Simple count")
count = count_objects("C:/Users/user/Desktop/apple.jpeg")
print(f"Count: {count}\n")

# Test 2: Me konfigurimin tuaj
print("Test 2: With custom config")
configure_detection(min_area=1200, peak_rel=0.55)
count = count_objects("C:/Users/user/Desktop/apple.jpeg", mode="watershed")
print(f"Count: {count}\n")

# Test 3: Analizë e plotë
print("Test 3: Full analysis")
result = analyze_image(
    "C:/Users/user/Desktop/apple.jpeg",
    mode="watershed",
    iterations=100,
    confidence_level=0.90,
    verbose=True
)
print(f"Count: {result['count']}")
print(f"CI: {result['confidence_interval']}")
print(f"Time: {result['processing_time']:.2f}s\n")

# Test 4: Me grupim
print("Test 4: With grouping")
result = analyze_image(
    "C:/Users/user/Desktop/apple.jpeg",
    iterations=555,
    do_group=True,
    verbose=False
)
groups = Counter(result['groups']['labels'])
print(f"Groups: {dict(groups)}")
print(f"Silhouette: {result['groups']['silhouette']:.3f}\n")

# Test 5: Me klasifikim
print("Test 5: With classification")
result = analyze_image(
    "C:/Users/user/Desktop/apple.jpeg",
    iterations=50,
    do_classify=True,
    verbose=False
)
patterns = Counter(l['label'] for l in result['labels'])
print(f"Patterns: {dict(patterns)}\n")

# Test 6: Kombinuar
print("Test 6: Combined")
result = analyze_image(
    "C:/Users/user/Desktop/apple.jpeg",
    iterations=100,
    do_group=True,
    do_classify=True,
    verbose=True
)
print(f"\nResults:")
print(f"  Count: {result['count']}")
print(f"  Groups: {result['groups']['k']}")
print(f"  Patterns: {len(set(l['label'] for l in result['labels']))}")