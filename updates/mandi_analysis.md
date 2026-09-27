# Why the Model Struggles with Mandi

Mandi is notoriously tricky for binary food classifiers (veg vs. non-veg) to get right. Based on the predictions from your dataset (where 4 out of 7 images were incorrectly flagged as `veg`), here are the main reasons why the model is getting confused:

### 1. The "Sea of Rice" Problem (High Rice-to-Meat Ratio)
Visually, a plate of Mandi is predominantly rice. The spiced yellow/brown rice takes up 70-90% of the pixels in the image. During training, the model likely learned to heavily associate large plates of grains/rice with vegetarian dishes (like plain biryani, pilaf, or mujadara). If the meat is small or buried, the model defaults to `veg`.

### 2. YOLO Segmentation Cropping
Your testing script uses `yolov8n-seg.pt` to crop the image to the main "food" object before passing it to the classifier. 
- If YOLO determines that the massive mound of rice is the primary food object, it might draw the bounding box/segmentation mask purely around the rice.
- In doing so, it might accidentally crop out the chicken or lamb resting on the edge of the plate. If the classifier only sees the cropped rice, it will naturally predict `veg`.

### 3. Meat Camouflage
The meat in Mandi is often slow-roasted or smoked, giving it a golden-brown or charred appearance that is very similar to roasted vegetables, fried potatoes, or nuts. Additionally, the meat is often covered in raisins, nuts, and caramelized onions, making it harder for the model to isolate the "meat" texture.

---

## 🔍 Let's look at some examples from your dataset:

### 🔴 Incorrectly Predicted as VEG 🥬
These images were predicted as `veg`. Notice how dominant the rice is compared to the meat, or how the meat blends in.
````carousel
![Mandi 1 (Predicted Veg: 86.9%)](/Users/mehdidar/.gemini/antigravity-ide/brain/63eea93e-4735-4752-89d5-3bb29cb93ade/mandi_images/images (1).jpg)
<!-- slide -->
![Mandi 2 (Predicted Veg: 79.7%)](/Users/mehdidar/.gemini/antigravity-ide/brain/63eea93e-4735-4752-89d5-3bb29cb93ade/mandi_images/images (2).jpg)
<!-- slide -->
![Mandi 4 (Predicted Veg: 65.6%)](/Users/mehdidar/.gemini/antigravity-ide/brain/63eea93e-4735-4752-89d5-3bb29cb93ade/mandi_images/images (4).jpg)
<!-- slide -->
![Mandi 7 (Predicted Veg: 55.3% - Uncertain)](/Users/mehdidar/.gemini/antigravity-ide/brain/63eea93e-4735-4752-89d5-3bb29cb93ade/mandi_images/ربع تيس مندي.webp)
````

### ✅ Correctly Predicted as NON-VEG 🥩
These images were correctly predicted. In these photos, the meat (chicken or lamb) is usually much more prominent, clearly visible, and takes up a larger portion of the frame.
````carousel
![Mandi 5 (Predicted Non-Veg: 75.6%)](/Users/mehdidar/.gemini/antigravity-ide/brain/63eea93e-4735-4752-89d5-3bb29cb93ade/mandi_images/images (5).jpg)
<!-- slide -->
![Mandi 6 (Predicted Non-Veg: 76.5%)](/Users/mehdidar/.gemini/antigravity-ide/brain/63eea93e-4735-4752-89d5-3bb29cb93ade/mandi_images/images.jpg)
<!-- slide -->
![Mandi 3 (Predicted Non-Veg: 57.0% - Uncertain)](/Users/mehdidar/.gemini/antigravity-ide/brain/63eea93e-4735-4752-89d5-3bb29cb93ade/mandi_images/images (3).jpg)
````

### 💡 How can we fix this?
1. **Turn off YOLO cropping:** Try running the script with the `--no_yolo` flag. This passes the entire raw image to the classifier, ensuring the meat doesn't get accidentally cropped out.
2. **More training data:** We would need to train the model on a lot more Mandi images specifically, forcing it to look for the meat even when overwhelmed by rice.
