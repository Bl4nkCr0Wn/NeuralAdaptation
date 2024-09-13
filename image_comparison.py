import cv2

def compare_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    ssim = cv2.SSIM(gray1, gray2, multichannel=False)

    return ssim

# Load images
image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

# Compare images
similarity = compare_images(image1, image2)

print("Similarity:", similarity)