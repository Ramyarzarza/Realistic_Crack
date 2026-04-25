from __future__ import annotations

from pathlib import Path

import cv2


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
INPUT_DIR = Path("samples")
OUTPUT_DIR = Path("samples_split")
GRID_ROWS = 4
GRID_COLS = 4


def iter_image_files(input_dir: Path) -> list[Path]:
	return sorted(
		file_path
		for file_path in input_dir.iterdir()
		if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS
	)


def split_image(image_path: Path, output_dir: Path, rows: int, cols: int) -> int:
	image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
	if image is None:
		raise ValueError(f"Could not read image: {image_path}")

	height, width = image.shape[:2]
	base_name = image_path.stem
	extension = image_path.suffix

	tile_count = 0

	for row_index in range(rows):
		start_y = (height * row_index) // rows
		end_y = (height * (row_index + 1)) // rows

		for col_index in range(cols):
			start_x = (width * col_index) // cols
			end_x = (width * (col_index + 1)) // cols

			tile = image[start_y:end_y, start_x:end_x]
			tile_name = f"{base_name}_r{row_index + 1:02d}_c{col_index + 1:02d}{extension}"
			tile_path = output_dir / tile_name
			if not cv2.imwrite(str(tile_path), tile):
				raise ValueError(f"Could not save tile: {tile_path}")
			tile_count += 1

	return tile_count


def main() -> None:
	if GRID_ROWS <= 0 or GRID_COLS <= 0:
		raise ValueError("GRID_ROWS and GRID_COLS must be positive integers")

	input_dir = INPUT_DIR.expanduser().resolve()
	output_dir = OUTPUT_DIR.expanduser().resolve()

	if not input_dir.exists() or not input_dir.is_dir():
		raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

	image_files = iter_image_files(input_dir)
	if not image_files:
		raise FileNotFoundError(f"No supported image files found in: {input_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)

	total_tiles = 0
	for image_path in image_files:
		total_tiles += split_image(image_path, output_dir, GRID_ROWS, GRID_COLS)

	print(
		f"Processed {len(image_files)} images from {input_dir}. "
		f"Saved {total_tiles} tiles to {output_dir}. "
		f"Grid: {GRID_ROWS}x{GRID_COLS}."
	)


if __name__ == "__main__":
	main()
