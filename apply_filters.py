#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import logging
from datetime import datetime
import sys
from PIL import Image

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Setup logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    def format(self, record):
        if record.levelname == 'INFO':
            color = Colors.GREEN
        elif record.levelname == 'WARNING':
            color = Colors.YELLOW
        elif record.levelname == 'ERROR':
            color = Colors.RED
        else:
            color = Colors.ENDC
        record.msg = f"{color}{record.msg}{Colors.ENDC}"
        return super().format(record)

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class PhotoFilter:
    def __init__(self):
        # Statistics
        self.stats = {
            'total_photos': 0,
            'processed_photos': 0,
            'failed_photos': 0
        }
        
        # Filter presets
        self.presets = {
            'vintage': {
                'description': 'Vintage film effect with warm tones',
                'function': self.apply_vintage_filter
            },
            'cool': {
                'description': 'Cool blue tones',
                'function': self.apply_cool_filter
            },
            'warm': {
                'description': 'Warm orange tones',
                'function': self.apply_warm_filter
            },
            'high_contrast': {
                'description': 'High contrast black and white',
                'function': self.apply_high_contrast_filter
            },
            'sepia': {
                'description': 'Classic sepia effect',
                'function': self.apply_sepia_filter
            },
            'blur': {
                'description': 'Soft blur effect',
                'function': self.apply_blur_filter
            },
            'sharpen': {
                'description': 'Sharpen image details',
                'function': self.apply_sharpen_filter
            },
            'emboss': {
                'description': 'Embossed effect',
                'function': self.apply_emboss_filter
            },
            'edge_detect': {
                'description': 'Edge detection effect',
                'function': self.apply_edge_detect_filter
            },
            'cartoon': {
                'description': 'Cartoon-like effect',
                'function': self.apply_cartoon_filter
            }
        }
    
    def print_header(self):
        """Print program header information"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Photo Filter Tool ==={Colors.ENDC}")
        print(f"{Colors.BLUE}Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    def print_footer(self):
        """Print program footer statistics"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Processing Complete ==={Colors.ENDC}")
        print(f"{Colors.BLUE}End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
        print("\nProcessing Statistics:")
        print(f"{Colors.GREEN}Total photos processed: {self.stats['processed_photos']}{Colors.ENDC}")
        print(f"{Colors.RED}Failed photos: {self.stats['failed_photos']}{Colors.ENDC}")
    
    def print_presets(self):
        """Print available filter presets"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}Available Filter Presets:{Colors.ENDC}")
        for name, info in self.presets.items():
            print(f"{Colors.GREEN}{name}:{Colors.ENDC} {info['description']}")
    
    def apply_vintage_filter(self, img, strength=0.5):
        """Apply vintage film effect"""
        # Convert to float32 for calculations
        img = img.astype(np.float32) / 255.0
        
        # Add warm tones
        img[:, :, 0] *= (1 + strength * 0.1)  # Red
        img[:, :, 1] *= (1 + strength * 0.05)  # Green
        img[:, :, 2] *= (1 - strength * 0.1)   # Blue
        
        # Add vignette effect
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/4)
        kernel_y = cv2.getGaussianKernel(rows, rows/4)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        img = img * (1 - strength * (1 - mask))
        
        # Add noise
        noise = np.random.normal(0, strength * 0.05, img.shape)
        img = img + noise
        
        # Clip values and convert back to uint8
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)
    
    def apply_cool_filter(self, img, strength=0.5):
        """Apply cool blue tones"""
        # Convert to float32 for calculations
        img = img.astype(np.float32) / 255.0
        
        # Add blue tones
        img[:, :, 0] *= (1 - strength * 0.1)  # Red
        img[:, :, 1] *= (1 - strength * 0.05)  # Green
        img[:, :, 2] *= (1 + strength * 0.1)   # Blue
        
        # Clip values and convert back to uint8
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)
    
    def apply_warm_filter(self, img, strength=0.5):
        """Apply warm orange tones"""
        # Convert to float32 for calculations
        img = img.astype(np.float32) / 255.0
        
        # Add warm tones
        img[:, :, 0] *= (1 + strength * 0.1)  # Red
        img[:, :, 1] *= (1 + strength * 0.05)  # Green
        img[:, :, 2] *= (1 - strength * 0.1)   # Blue
        
        # Clip values and convert back to uint8
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)
    
    def apply_high_contrast_filter(self, img, strength=0.5):
        """Apply high contrast black and white effect"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            11, 2
        )
        
        # Blend with original
        return cv2.addWeighted(gray, 1 - strength, thresh, strength, 0)
    
    def apply_sepia_filter(self, img, strength=0.5):
        """Apply sepia effect"""
        # Convert to float32 for calculations
        img = img.astype(np.float32) / 255.0
        
        # Sepia matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Apply sepia effect
        sepia = cv2.transform(img, sepia_matrix)
        
        # Blend with original
        return (cv2.addWeighted(img, 1 - strength, sepia, strength, 0) * 255).astype(np.uint8)
    
    def apply_blur_filter(self, img, strength=0.5):
        """Apply blur effect"""
        kernel_size = int(5 + strength * 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def apply_sharpen_filter(self, img, strength=0.5):
        """Apply sharpen effect"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) * strength
        return cv2.filter2D(img, -1, kernel)
    
    def apply_emboss_filter(self, img, strength=0.5):
        """Apply emboss effect"""
        kernel = np.array([[-2,-1, 0],
                          [-1, 1, 1],
                          [ 0, 1, 2]]) * strength
        return cv2.filter2D(img, -1, kernel)
    
    def apply_edge_detect_filter(self, img, strength=0.5):
        """Apply edge detection effect"""
        edges = cv2.Canny(img, 100, 200)
        return cv2.addWeighted(img, 1 - strength, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), strength, 0)
    
    def apply_cartoon_filter(self, img, strength=0.5):
        """Apply cartoon effect"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply median blur
        gray = cv2.medianBlur(gray, 5)
        
        # Detect edges
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            9, 9
        )
        
        # Apply bilateral filter
        color = cv2.bilateralFilter(img, 9, 250, 250)
        
        # Combine edges and color
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        # Blend with original
        return cv2.addWeighted(img, 1 - strength, cartoon, strength, 0)
    
    def apply_filter(self, img, filter_name, strength=0.5):
        """Apply selected filter to image"""
        if filter_name not in self.presets:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        return self.presets[filter_name]['function'](img, strength)
    
    def process_photo(self, img_path, output_dir, filter_name, strength=0.5, dry_run=False):
        """Process a single photo"""
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"❌ Cannot read image: {img_path}")
                return False
            
            # Apply filter
            filtered_img = self.apply_filter(img, filter_name, strength)
            
            # Generate output filename
            output_filename = f"{Path(img_path).stem}_{filter_name}{Path(img_path).suffix}"
            output_path = os.path.join(output_dir, output_filename)
            
            if dry_run:
                logger.info(f"📋 Preview: {os.path.basename(img_path)} -> {output_filename}")
            else:
                # Save filtered image
                cv2.imwrite(output_path, filtered_img)
                logger.info(f"✅ Processed: {os.path.basename(img_path)} -> {output_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {img_path}: {str(e)}")
            return False
    
    def process_directory(self, input_dir, output_dir, filter_name, strength=0.5, dry_run=False):
        """Process all photos in directory"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png'}
        
        # Get all image files
        image_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        self.stats['total_photos'] = len(image_files)
        if self.stats['total_photos'] == 0:
            logger.warning(f"{Colors.YELLOW}⚠️ No image files found{Colors.ENDC}")
            return
        
        logger.info(f"{Colors.BLUE}📁 Found {self.stats['total_photos']} image files{Colors.ENDC}")
        
        # Process files
        for i, img_path in enumerate(image_files, 1):
            # Show progress
            progress = (i / self.stats['total_photos']) * 100
            sys.stdout.write(f"\r{Colors.BLUE}Processing: {progress:.1f}% ({i}/{self.stats['total_photos']}){Colors.ENDC}")
            sys.stdout.flush()
            
            if self.process_photo(img_path, output_dir, filter_name, strength, dry_run):
                self.stats['processed_photos'] += 1
            else:
                self.stats['failed_photos'] += 1
        
        print("\n")  # New line to avoid progress bar overlap

def main():
    parser = argparse.ArgumentParser(description='Photo Filter Tool')
    parser.add_argument('input_dir', help='Input directory path')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('filter', help='Filter preset name')
    parser.add_argument('--dry-run', action='store_true', help='Preview operations without executing')
    parser.add_argument('--strength', type=float, default=0.5, help='Filter strength (0.0-1.0)')
    parser.add_argument('--list-presets', action='store_true', help='List available filter presets')
    args = parser.parse_args()
    
    # Create filter instance
    filter_tool = PhotoFilter()
    
    # List presets if requested
    if args.list_presets:
        filter_tool.print_presets()
        return
    
    # Check if filter exists
    if args.filter not in filter_tool.presets:
        logger.error(f"❌ Unknown filter: {args.filter}")
        logger.info("Use --list-presets to see available filters")
        return
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"❌ Input directory does not exist: {args.input_dir}")
        return
        
    # Check if input directory is readable
    if not os.access(args.input_dir, os.R_OK):
        logger.error(f"❌ Input directory cannot be accessed (permission issue): {args.input_dir}")
        return
    
    filter_tool.print_header()
    filter_tool.process_directory(args.input_dir, args.output_dir, args.filter, args.strength, args.dry_run)
    filter_tool.print_footer()

if __name__ == '__main__':
    main() 