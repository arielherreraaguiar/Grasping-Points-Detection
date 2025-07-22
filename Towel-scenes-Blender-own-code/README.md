# Towel Scene Generator

This script generates a dataset of realistic towel scenes rendered in Blender. Towels can appear in different configurations: flat, messy (with folds and wrinkles), or folded in half. The output consists of `.png` images.

## Requirements

- Blender 3.6.5 (or compatible)
- Python (bundled with Blender)
- A terminal or shell environment

## How to Run

To generate the towel dataset, open a terminal and run the following command:

```bash
~/blender-3.6.5/blender --background --python generate_towel_dataset.py
```

This will launch Blender in background mode (no user interface) and execute the `generate_towel_dataset.py` script.

## Output

Rendered images will be saved to the following directory:

```
/home/ariel/Downloads/towel-scenes/towel_output/
```

You can change the output directory by editing the `output_dir` variable at the top of the script.

## Customization

You can customize the script to:
- Change the number of generated scenes
- Adjust towel types (flat, messy, half_folded)
- Modify camera angles, lighting, or resolution
