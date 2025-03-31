# Smart Photo Management Toolkit

A Python toolkit for intelligent photo management, featuring photo quality classification, scene similarity grouping, and photo filtering capabilities.

## Features

1. **Photo Quality Classification**
   - Automatically classifies photos based on focus and composition quality
   - Uses OpenCV for image analysis
   - Creates organized output directories for different quality levels

2. **Scene Similarity Grouping**
   - Groups similar photos based on scene content
   - Uses SIFT and FLANN for feature matching
   - Renames files with scene numbers for easy organization

3. **Photo Filtering**
   - Apply various artistic filters to photos
   - Supports multiple preset effects
   - Adjustable filter strength
   - Batch processing capability

## Installation Requirements

### System Requirements
- Python 3.8 or higher
- macOS, Linux, or Windows
- Sufficient disk space for photo processing

### Dependencies
```
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
ImageHash>=4.3.1
```

## Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd photo-management-toolkit
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Photo Quality Classification

Classify photos based on focus and composition quality:

```bash
python classify_photos.py /path/to/photos /path/to/output
```

Options:
- `--dry-run`: Preview operations without executing
- `--threshold N`: Adjust classification threshold (default: 0.7)

### 2. Scene Similarity Grouping

Group similar photos based on scene content:

```bash
python group_similar_scenes.py /path/to/photos /path/to/output
```

Options:
- `--dry-run`: Preview operations without executing
- `--threshold N`: Adjust similarity threshold (default: 0.7)

### 3. Photo Filtering

Apply artistic filters to photos:

```bash
python apply_filters.py /path/to/photos /path/to/output filter_name
```

Options:
- `--dry-run`: Preview operations without executing
- `--strength N`: Adjust filter strength (0.0-1.0, default: 0.5)
- `--list-presets`: List available filter presets

Available filter presets:
- `vintage`: Vintage film effect with warm tones
- `cool`: Cool blue tones
- `warm`: Warm orange tones
- `high_contrast`: High contrast black and white
- `sepia`: Classic sepia effect
- `blur`: Soft blur effect
- `sharpen`: Sharpen image details
- `emboss`: Embossed effect
- `edge_detect`: Edge detection effect
- `cartoon`: Cartoon-like effect

## Output Structure

### Classified Photos
```
output/
├── high_quality/
├── medium_quality/
└── low_quality/
```

### Grouped Scenes
```
output/
├── scene_001_001.jpg
├── scene_001_002.jpg
├── scene_002_001.jpg
└── scene_002_002.jpg
```

### Filtered Photos
```
output/
├── original_vintage.jpg
├── original_cool.jpg
└── original_warm.jpg
```

## Important Notes

1. **File Permissions**
   - Ensure you have read/write permissions for input/output directories
   - For SD card operations, ensure proper mounting and permissions

2. **Storage Space**
   - Ensure sufficient disk space for processing
   - Original files are preserved unless explicitly deleted

3. **File Formats**
   - Supported formats: JPG, JPEG, PNG
   - RAW formats are preserved during processing

4. **SD Card Usage**
   - Always safely eject SD cards after use
   - Use the correct mount path for your system

## Common Issues

1. **File Permission Errors**
   - Solution: Check and adjust file/directory permissions
   - For SD cards: Ensure proper mounting and access rights

2. **Memory Issues**
   - Solution: Process photos in smaller batches
   - Close other memory-intensive applications

3. **Unsupported File Formats**
   - Solution: Convert unsupported formats to JPG/PNG before processing
   - Use appropriate file conversion tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

[Your Name]

## Changelog

### v1.0.0
- Initial release
- Photo quality classification
- Scene similarity grouping
- Photo filtering capabilities 