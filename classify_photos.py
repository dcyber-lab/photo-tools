#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import logging
from datetime import datetime
import sys

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

class PhotoClassifier:
    def __init__(self):
        # Create output directories
        self.categories = {
            'good': 'good_photos',
            'out_of_focus': 'out_of_focus',
            'poor_composition': 'poor_composition',
            'blurry': 'blurry',
            'unknown': 'unknown'
        }
        
        # Set thresholds
        self.focus_threshold = 100  # Laplacian variance threshold
        self.blur_threshold = 50    # Blur threshold
        
        # Statistics
        self.stats = {category: 0 for category in self.categories}
        
    def create_directories(self, base_dir):
        """Create classification directories"""
        for category in self.categories.values():
            os.makedirs(os.path.join(base_dir, category), exist_ok=True)
    
    def print_header(self):
        """Print program header information"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Photo Quality Classification Tool ==={Colors.ENDC}")
        print(f"{Colors.BLUE}Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    def print_footer(self):
        """Print program footer statistics"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Processing Complete ==={Colors.ENDC}")
        print(f"{Colors.BLUE}End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
        print("\nClassification Statistics:")
        for category, count in self.stats.items():
            color = Colors.GREEN if category == 'good' else Colors.YELLOW if category == 'unknown' else Colors.RED
            print(f"{color}{self.categories[category]}: {count} photos{Colors.ENDC}")
    
    def calculate_laplacian_variance(self, img):
        """Calculate Laplacian variance to evaluate focus quality"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def analyze_composition(self, img):
        """Analyze composition quality"""
        height, width = img.shape[:2]
        
        # Calculate image center point
        center_x = width // 2
        center_y = height // 2
        
        # Calculate edge region average brightness
        edge_width = width // 4
        edge_height = height // 4
        
        # Extract edge regions
        top_edge = img[0:edge_height, :]
        bottom_edge = img[height-edge_height:height, :]
        left_edge = img[:, 0:edge_width]
        right_edge = img[:, width-edge_width:width]
        
        # Calculate edge region average brightness
        edge_brightness = np.mean([np.mean(top_edge), np.mean(bottom_edge),
                                 np.mean(left_edge), np.mean(right_edge)])
        
        # Calculate center region average brightness
        center_region = img[center_y-edge_height:center_y+edge_height,
                          center_x-edge_width:center_x+edge_width]
        center_brightness = np.mean(center_region)
        
        # Calculate brightness contrast
        brightness_contrast = abs(center_brightness - edge_brightness)
        
        return brightness_contrast
    
    def analyze_image(self, img):
        """Analyze image quality"""
        # Calculate focus score
        focus_score = self.calculate_laplacian_variance(img)
        
        # Calculate composition score
        composition_score = self.analyze_composition(img)
        
        # Check blur
        is_blurry = focus_score < self.blur_threshold
        
        return {
            'focus_score': focus_score,
            'composition_score': composition_score,
            'is_blurry': is_blurry
        }
    
    def check_file(self, img_path):
        """Check if file is accessible"""
        if not os.path.exists(img_path):
            logger.error(f"❌ File does not exist: {img_path}")
            return False
            
        if not os.access(img_path, os.R_OK):
            logger.error(f"❌ File cannot be read (permission issue): {img_path}")
            return False
            
        try:
            # Try to open file with PIL
            with Image.open(img_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.error(f"❌ File format error or corrupted: {img_path}, Error: {str(e)}")
            return False
    
    def classify_photo(self, img_path, output_dir):
        """Classify a single photo"""
        try:
            # Check file
            if not self.check_file(img_path):
                return 'unknown'
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"❌ OpenCV cannot read image: {img_path}")
                return 'unknown'
            
            # Check if image is empty
            if img.size == 0:
                logger.error(f"❌ Image is empty: {img_path}")
                return 'unknown'
            
            # Analyze image
            analysis = self.analyze_image(img)
            
            # Classify based on analysis
            if analysis['is_blurry']:
                return 'blurry'
            elif analysis['focus_score'] < self.focus_threshold:
                return 'out_of_focus'
            elif analysis['composition_score'] < 30:  # Composition threshold
                return 'poor_composition'
            else:
                return 'good'
                
        except Exception as e:
            logger.error(f"❌ Error processing image {img_path}: {str(e)}")
            return 'unknown'
    
    def process_directory(self, input_dir, output_dir, dry_run=False):
        """Process all photos in directory"""
        self.create_directories(output_dir)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.arw', '.cr2', '.nef'}
        
        # Get all files to process
        all_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    all_files.append(os.path.join(root, file))
        
        total_files = len(all_files)
        if total_files == 0:
            logger.warning(f"{Colors.YELLOW}⚠️ No image files found{Colors.ENDC}")
            return
        
        logger.info(f"{Colors.BLUE}📁 Found {total_files} image files{Colors.ENDC}")
        
        # Process files
        for i, img_path in enumerate(all_files, 1):
            # Show progress
            progress = (i / total_files) * 100
            sys.stdout.write(f"\r{Colors.BLUE}Processing: {progress:.1f}% ({i}/{total_files}){Colors.ENDC}")
            sys.stdout.flush()
            
            category = self.classify_photo(img_path, output_dir)
            self.stats[category] += 1
            
            # Build target path
            target_dir = os.path.join(output_dir, self.categories[category])
            target_path = os.path.join(target_dir, os.path.basename(img_path))
            
            if dry_run:
                logger.info(f"📋 Preview: {os.path.basename(img_path)} -> {self.categories[category]}")
            else:
                try:
                    shutil.copy2(img_path, target_path)
                    logger.info(f"✅ Copied: {os.path.basename(img_path)} -> {self.categories[category]}")
                except Exception as e:
                    logger.error(f"❌ Copy failed: {os.path.basename(img_path)}: {str(e)}")
        
        print("\n")  # New line to avoid progress bar overlap

def main():
    parser = argparse.ArgumentParser(description='Photo Quality Classification Tool')
    parser.add_argument('input_dir', help='Input directory path')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('--dry-run', action='store_true', help='Preview operations without executing')
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"❌ Input directory does not exist: {args.input_dir}")
        return
        
    # Check if input directory is readable
    if not os.access(args.input_dir, os.R_OK):
        logger.error(f"❌ Input directory cannot be accessed (permission issue): {args.input_dir}")
        return
    
    classifier = PhotoClassifier()
    classifier.print_header()
    classifier.process_directory(args.input_dir, args.output_dir, args.dry_run)
    classifier.print_footer()

if __name__ == '__main__':
    main() 