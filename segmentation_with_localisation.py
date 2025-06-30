import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os


segmentation_model_path = "./segmentation_model/weights/best.pt"
segmentation_model = YOLO(segmentation_model_path)


img_path = "./brick-veg.jpg"


output_folder = "segmentation_outputs"
segmented_portions_folder = "segmented_portions"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(segmented_portions_folder, exist_ok=True)


def segment_image(image_np):

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    results = segmentation_model.predict(source=image_rgb, conf=0.3, save=False)

    segmented_image = results[0].plot()

    return results, segmented_image


def save_individual_masks_and_portions(
    results, original_image, output_folder, portions_folder
):
    masks = results[0].masks.data.cpu().numpy()
    image_height, image_width = original_image.shape[:2]

    for i, mask in enumerate(masks):

        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_image = np.zeros_like(original_image, dtype=np.uint8)

        mask_color = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        )

        mask_image[resized_mask == 1] = mask_color

        mask_output_path = os.path.join(output_folder, f"mask_{i + 1}.png")
        cv2.imwrite(mask_output_path, mask_image)
        print(f"Mask {i + 1} saved at {mask_output_path}")

        segmented_portion = cv2.bitwise_and(
            original_image, original_image, mask=(resized_mask * 255)
        )

        portion_output_path = os.path.join(
            portions_folder, f"segmented_portion_{i + 1}.png"
        )
        cv2.imwrite(portion_output_path, segmented_portion)
        print(f"Segmented portion {i + 1} saved at {portion_output_path}")


if __name__ == "__main__":

    image = Image.open(img_path)
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    print("Running segmentation...")
    results, segmented_image_with_overlay = segment_image(image_np)

    overlay_output_path = os.path.join(
        output_folder, "segmented_image_with_overlay.png"
    )
    cv2.imwrite(overlay_output_path, segmented_image_with_overlay)
    print(f"Segmented image with overlay saved at {overlay_output_path}")

    save_individual_masks_and_portions(
        results, image_np, output_folder, segmented_portions_folder
    )

    cv2.destroyAllWindows()
