#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import logging
from datetime import datetime
import sys
import shutil
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

class FaceDetector:
    def __init__(self):
        # Initialize face detection cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create output directories
        self.categories = {
            'no_faces': 'no_faces',
            'single_face': 'single_face',
            'multiple_faces': 'multiple_faces',
            'face_thumbnails': 'face_thumbnails'
        }
        
        # Statistics
        self.stats = {
            'total_photos': 0,
            'processed_photos': 0,
            'no_faces': 0,
            'single_face': 0,
            'multiple_faces': 0,
            'total_faces': 0
        }
        
        # Thumbnail settings
        self.thumbnail_size = (200, 200)
    
    def print_header(self):
        """Print program header information"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Photo Face Detection Tool ==={Colors.ENDC}")
        print(f"{Colors.BLUE}Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    def print_footer(self):
        """Print program footer statistics"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Processing Complete ==={Colors.ENDC}")
        print(f"{Colors.BLUE}End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
        print("\nDetection Statistics:")
        print(f"{Colors.GREEN}Total photos processed: {self.stats['processed_photos']}{Colors.ENDC}")
        print(f"{Colors.GREEN}Photos with no faces: {self.stats['no_faces']}{Colors.ENDC}")
        print(f"{Colors.GREEN}Photos with single face: {self.stats['single_face']}{Colors.ENDC}")
        print(f"{Colors.GREEN}Photos with multiple faces: {self.stats['multiple_faces']}{Colors.ENDC}")
        print(f"{Colors.GREEN}Total faces detected: {self.stats['total_faces']}{Colors.ENDC}")
    
    def create_directories(self, base_dir):
        """Create output directories"""
        for category in self.categories.values():
            os.makedirs(os.path.join(base_dir, category), exist_ok=True)
    
    def detect_faces(self, img):
        """Detect faces in image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def extract_face_thumbnail(self, img, face):
        """Extract and resize face region"""
        x, y, w, h = face
        face_img = img[y:y+h, x:x+w]
        return cv2.resize(face_img, self.thumbnail_size)
    
    def save_face_thumbnails(self, img, faces, output_dir, base_name):
        """Save face thumbnails"""
        for i, face in enumerate(faces):
            thumbnail = self.extract_face_thumbnail(img, face)
            thumbnail_path = os.path.join(
                output_dir,
                self.categories['face_thumbnails'],
                f"{base_name}_face_{i+1}.jpg"
            )
            cv2.imwrite(thumbnail_path, thumbnail)
    
    def process_photo(self, img_path, output_dir, dry_run=False):
        """Process a single photo"""
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"❌ Cannot read image: {img_path}")
                return
            
            # Detect faces
            faces = self.detect_faces(img)
            num_faces = len(faces)
            
            # Update statistics
            self.stats['total_faces'] += num_faces
            if num_faces == 0:
                self.stats['no_faces'] += 1
                category = 'no_faces'
            elif num_faces == 1:
                self.stats['single_face'] += 1
                category = 'single_face'
            else:
                self.stats['multiple_faces'] += 1
                category = 'multiple_faces'
            
            # Build target path
            target_dir = os.path.join(output_dir, self.categories[category])
            target_path = os.path.join(target_dir, os.path.basename(img_path))
            
            if dry_run:
                logger.info(f"📋 Preview: {os.path.basename(img_path)} -> {category} ({num_faces} faces)")
            else:
                # Copy photo to category directory
                shutil.copy2(img_path, target_path)
                logger.info(f"✅ Processed: {os.path.basename(img_path)} -> {category} ({num_faces} faces)")
                
                # Save face thumbnails if faces detected
                if num_faces > 0:
                    self.save_face_thumbnails(img, faces, output_dir, Path(img_path).stem)
            
        except Exception as e:
            logger.error(f"❌ Error processing {img_path}: {str(e)}")
    
    def process_directory(self, input_dir, output_dir, dry_run=False):
        """Process all photos in directory"""
        self.create_directories(output_dir)
        
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
            
            self.process_photo(img_path, output_dir, dry_run)
            self.stats['processed_photos'] += 1
        
        print("\n")  # New line to avoid progress bar overlap

def main():
    parser = argparse.ArgumentParser(description='Photo Face Detection Tool')
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
    
    detector = FaceDetector()
    detector.print_header()
    detector.process_directory(args.input_dir, args.output_dir, args.dry_run)
    detector.print_footer()

if __name__ == '__main__':
    main() 