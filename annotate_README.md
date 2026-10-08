# Manual Optic Disc Annotation Workflow

This guide describes how to annotate optic discs using GIMP, export PNG masks, and convert them into YOLO segmentation labels. Project folders are expressed relative to the repository root; the GIMP configuration derives its absolute image path from an anonymized repository-root placeholder. Start the notebook from that root.

## 1. Install and configure GIMP and the plugin

Use GIMP 3.0 or later and the [GIMP Mask Plugin](https://github.com/Pauwit/GIMP_mask_plugin/tree/main). Follow the repository's installation README as the reference for installation and configuration.

Installation checklist:

1. Locate the GIMP plug-ins directory (`%APPDATA%\GIMP\3.0\plug-ins\` on Windows or `~/.config/GIMP/3.0/plug-ins/` on Linux).
2. Create a `save_mask_and_next` subfolder and put the plugin script inside it as `save_mask_and_next.py`. Make it executable on Linux/macOS.
3. Edit `IMG_DIR` at the top of the script to point to the **exact folder containing the source PNG images to annotate**. For example:

   ```python
   # Replace /path/to/repository with the absolute path to your repository root.
   REPO_ROOT = "/path/to/repository"
   IMG_DIR = os.path.join(REPO_ROOT, "datasets", "images")
   ```

4. Set `IMG_DIR` to an absolute path derived from your repository root: GIMP does not necessarily start in that directory. Check the optional opacity, state-file, and debug-file settings. Restart GIMP after installing or editing the plugin.
5. Optionally assign a shortcut to **Save Mask and Next** through **Edit > Keyboard Shortcuts**.

## 2. Understand where masks are saved

The plugin automatically creates a **sibling folder** by appending `_masks` to the source image folder path. Each exported PNG uses the source image stem followed by `_mask.png` (singular).

For example, with `IMG_DIR` pointing to `datasets/images` under the repository root:

```text
repository/
└── datasets/
    ├── images/
    │   ├── nomimg.png
    │   └── another_image.png
    └── images_masks/
        ├── nomimg_mask.png
        └── another_image_mask.png
```

No export-path or naming changes to the plugin are required. Keep source PNG images in `IMG_DIR`; collect the generated masks from the separate sibling folder.

This behavior is documented in the [plugin README](https://github.com/Pauwit/GIMP_mask_plugin/blob/main/README.md) and confirmed by the [plugin script](https://github.com/Pauwit/GIMP_mask_plugin/blob/main/save_mask_and_next.py).

## 3. Annotate each image

1. Open the first source image in GIMP.
2. Invoke **File > Save Mask and Next** to initialize the first `Mask` layer as described in the plugin README.
3. Select the `Mask` layer and use the **Paintbrush** tool with **white** as the foreground color.
4. Paint and fill the entire optic disc region. Follow its visible boundary carefully; do not draw only an outline. Keep the background black and exclude surrounding retinal tissue.
5. Zoom in to refine the edge, then inspect the complete mask for gaps, stray strokes, or unwanted regions.
6. Invoke **File > Save Mask and Next** (or your assigned shortcut). The plugin sets the `Mask` layer opacity to 100%, flattens the working image, and saves `nomimg_mask.png` in the sibling `images_masks` folder. Keep the full-size, black-filled `Mask` layer visible above the source image so the exported result contains only the white optic disc on black.
7. After saving, the plugin closes the current working image, opens the next source PNG in alphabetical order, and creates a fresh black `Mask` layer. Repeat the painting and saving steps for each image. On the final image, verify that its mask exists: the current script saves it before looking for a next image, but may report an error when no next file is available.

Before annotating the full collection, save one test mask with the plugin and reopen it to verify its filename, location, dimensions, and black/white contents.

## 4. Convert PNG masks to YOLO labels

Use the dataset conversion notebook referred to as **new_dataset**.

1. Collect and verify all exported `*_mask.png` files.
2. Open the notebook and run the imports required by the conversion cell (`Path`, OpenCV, and NumPy).
3. Run the notebook with the repository root as its working directory. Set the input and output folders relative to that root:

   ```python
   masks_dir = Path("datasets/images_masks")
   labels_dir = Path("datasets/labels")
   ```
4. The notebook currently selects `*.png`. In the dedicated mask folder this selects the exported masks; you can use the more specific pattern:

   ```python
   for mask_path in masks_dir.glob("*_mask.png"):
   ```

5. Ensure the output label uses the **source image stem**, without the `_mask` suffix. The current notebook uses the mask stem directly, so adapt its filename construction:

   ```python
   image_stem = mask_path.stem.removesuffix("_mask")
   label_path = labels_dir / f"{image_stem}.txt"
   ```

6. Confirm `CLASS_ID = 0` corresponds to the optic disc class in the dataset configuration, then run the conversion cell.

The current conversion thresholds grayscale masks at 127, extracts external contours, and retains the largest contour because one optic disc is expected per image. It writes normalized polygon coordinates in YOLO **segmentation** format:

```text
class_id x1 y1 x2 y2 ... xn yn
```

Coordinates are normalized by image width and height. A mask with no contour or fewer than three contour points produces an empty label file. This is expected for a deliberately all-black mask when no optic disc is visible; otherwise, review the annotation.

Expected image-to-label pairing:

```text
nomimg.png        -> nomimg_mask.png        -> nomimg.txt
```

## 5. Final quality checks

- Every annotated source image has one correctly named PNG mask and one matching YOLO `.txt` label.
- Masks have the same dimensions as their source images, with a filled white optic disc on black when visible, or an entirely black mask when no optic disc is visible.
- Source images and exported masks have not been confused during conversion.
- Nonempty labels contain class `0` followed by normalized coordinate pairs between 0 and 1.
- Overlay a sample of the converted polygons on the original images to confirm that the annotations align with the optic discs.

## Annotate images with no visible optic disc

If no optic disc is visible, keep the image as a negative example and explicitly annotate it as containing no object.

1. In GIMP, leave the entire `Mask` layer black, without any white brush strokes.
2. Use **File > Save Mask and Next** to save the all-black mask as `nomimg_mask.png` in the sibling mask folder (for example, `datasets/images_masks`).
3. Run the notebook conversion. Since the mask has no contour, the conversion cell creates an empty `.txt` label.
4. Ensure the final label matches the source image name: `nomimg.png` must be paired with `nomimg.txt`, removing the `_mask` suffix from the label filename as described above.

The `.txt` file must be **completely empty**. Do not write a class ID, coordinates, a placeholder polygon, or a background class. Ultralytics YOLO accepts empty labels as images containing no annotated object ([official label-loading implementation](https://docs.ultralytics.com/reference/data/utils/#ultralytics.data.utils.verify_image_label)).

```text
Source image: datasets/images/nomimg.png
Mask:         datasets/images_masks/nomimg_mask.png (entirely black)
YOLO label:   datasets/labels/nomimg.txt (empty file)
```

An empty annotation means the image has been reviewed and no optic disc is visible. Do not use an empty label for an image that has not been annotated yet. If part of the disc is visible, apply a consistent partial-visibility annotation rule across the dataset rather than automatically marking the image as negative.

During quality checks, accept deliberately all-black masks and their empty labels for these negative examples. Investigate unexpected empty labels in images where the optic disc is visible.


