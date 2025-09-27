from architecture import UNet
import matplotlib.pyplot as plt
from kornia.color import rgb_to_lab, lab_to_rgb
import torch
from PIL import Image
import numpy as np

model = UNet().to('cuda')
# model.load_state_dict(torch.load('checkpoints/epoch_011.pth', map_location='cuda')['model'])
model.eval()
image_path = 'results/test.jpg'

# print number of parameters in the model
# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters in the model: {num_params}")

# Load the image properly
img = Image.open(image_path).convert('RGB')
img_array = np.array(img) / 255.0  # Normalize to [0, 1]
img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float().to('cuda')

# Convert RGB to LAB color space
lab_tensor = rgb_to_lab(img_tensor)

# Extract L channel and normalize
L = lab_tensor[:, 0:1, :, :]  # Shape: [1, 1, H, W]
L_normalized = ((L / 100.0) - 0.449) / 0.226  # Normalize to [0, 1]

# Generate the ab channels
with torch.no_grad():
    ab_pred = model(L_normalized)

# Combine L and predicted ab channels
combined_lab = torch.cat([L, ab_pred * 128.0], dim=1)  # Scale ab to [-128, 127]

# Convert back to RGB
colorized_rgb = lab_to_rgb(combined_lab)

# Save the colorized image
# output_path = f'colorized_{image_path}'
# colorized_np = colorized_rgb[0].permute(1, 2, 0).detach().cpu().numpy()
# colorized_np = np.clip(colorized_np, 0, 1)
# colorized_img = Image.fromarray((colorized_np * 255).astype(np.uint8))
# colorized_img = colorized_img.resize(img.size, resample=Image.LANCZOS)
# colorized_img.save(output_path, quality=100, subsampling=0)

# save original image as black and white
# bw_img = img.convert('L')
# bw_img.save('original_bw.jpg')

# Display results - simplified to show only 3 images
# plt.figure(figsize=(12, 4))  # Reduced width for better proportions

# 1. Left: L channel (grayscale input)
plt.subplot(1, 3, 1)
plt.imshow(L[0, 0].cpu().numpy(), cmap='gray', aspect='equal')
plt.title('L Channel (Input)')
plt.axis('off')

# 2. Middle: Colorized prediction
plt.subplot(1, 3, 2)
plt.imshow(colorized_rgb[0].permute(1, 2, 0).cpu().numpy(), aspect='equal')
plt.title('Colorized Prediction')
plt.axis('off')

# 3. Right: Ground truth
plt.subplot(1, 3, 3)
plt.imshow(img_array, aspect='equal')
plt.title('Ground Truth (Original)')
plt.axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.05)  # Reduce spacing between subplots
plt.show()