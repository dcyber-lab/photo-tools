# Smart Photo Management Toolkit

A Python toolkit for intelligent photo management with two main features:
1. Photo Quality Classification
2. Scene Similarity Grouping

## Features

### 1. Photo Quality Classification (classify_photos.py)
- Automatic photo quality detection
- Classification criteria:
  - Focus quality
  - Composition quality
  - Blur detection
- Supports multiple image formats (JPG, JPEG, PNG, ARW, CR2, NEF)
- Provides dry-run mode for preview

### 2. Scene Similarity Grouping (group_similar_scenes.py)
- Uses SIFT algorithm for scene recognition
- Automatic file renaming
- Naming convention: `scene_XXX_YYY.ext`
  - XXX: Scene number (starting from 001)
  - YYY: Image sequence number within the same scene (starting from 001)
- Adjustable similarity threshold
- Supports JPG, JPEG, PNG formats

## Installation Requirements

### System Requirements
- Python 3.9 or higher
- macOS/Linux/Windows

### Dependencies
```
opencv-python>=4.5.0
numpy>=1.19.0
Pillow>=8.0.0
```

## Installation Steps

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Photo Quality Classification

Preview operations:
```bash
python classify_photos.py /path/to/input/directory /path/to/output/directory --dry-run
```

Execute classification:
```bash
python classify_photos.py /path/to/input/directory /path/to/output/directory
```

### 2. Scene Similarity Grouping

Preview operations:
```bash
python group_similar_scenes.py /path/to/input/directory --dry-run
```

Execute grouping:
```bash
python group_similar_scenes.py /path/to/input/directory
```

Adjust similarity threshold:
```bash
python group_similar_scenes.py /path/to/input/directory --threshold 0.8
```

## Output Description

### Photo Quality Classification Output Directories
- `good_photos/`: Well-focused and well-composed photos
- `out_of_focus/`: Photos with focus issues
- `poor_composition/`: Photos with composition issues
- `blurry/`: Blurry photos
- `unknown/`: Photos that cannot be analyzed

### Scene Grouping Output
- Files will be renamed to: `scene_XXX_YYY.ext`
- Example: `scene_001_001.jpg`, `scene_001_002.jpg`

## Important Notes

1. File Permissions
   - Ensure read permissions for input directory
   - Ensure write permissions for output directory

2. Storage Space
   - Ensure sufficient disk space
   - Recommended to use dry-run mode before processing large batches

3. File Formats
   - Supported formats: JPG, JPEG, PNG, ARW, CR2, NEF
   - Ensure files are not corrupted

4. SD Card Usage
   - Ensure SD card is properly mounted
   - Check if SD card is locked
   - Ensure sufficient read permissions

## Common Issues

1. Unable to Read Images
   - Check file permissions
   - Verify supported file format
   - Validate file integrity

2. Poor Grouping Results
   - Adjust similarity threshold
   - Ensure good image quality
   - Check for image corruption

3. Permission Issues
   - Using sudo (not recommended)
   - Modify file permissions
   - Check directory permissions

## Contributing

Issues and Pull Requests are welcome to improve this project.

## License

MIT License

## Author

[Your Name]

## Changelog

### v1.0.0
- Initial release
- Implemented photo quality classification
- Implemented scene similarity grouping 