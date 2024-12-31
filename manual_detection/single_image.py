import cv2
# Read image
img = cv2.imread("/home/yuming/Documents/mt_data/MT10_30min_200x_1500_138_146pm/MT10_30min_200x_1500_138_146pm_t1500.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.normalize(
    img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
img = cv2.bilateralFilter(img, 15, 75, 75) 

# Modify the parameters of MSER
mser = cv2.MSER_create(delta=2, min_area=200, max_area=5000, max_variation=1, min_diversity=0.1)
# Detect regions
regions, _ = mser.detectRegions(gray)
print(len(regions))
# Filter regions by area
filtered_regions = [p for p in regions if len(p) > 10]
# Draw the regions on the image
for p in filtered_regions:
    x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# Display the result
cv2.imshow("MSER regions", img)
cv2.waitKey(0)