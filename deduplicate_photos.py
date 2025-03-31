#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import logging
from datetime import datetime
import sys
import imagehash
from PIL import Image
import hashlib
from collections import defaultdict

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

class PhotoDeduplicator:
    def __init__(self):
        # Initialize SIFT feature detector
        self.sift = cv2.SIFT_create()
        
        # Initialize FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Set thresholds
        self.hash_threshold = 5  # Perceptual hash difference threshold
        self.feature_threshold = 0.7  # Feature matching threshold
        
        # Statistics
        self.stats = {
            'total_photos': 0,
            'processed_photos': 0,
            'duplicate_groups': 0,
            'duplicate_photos': 0,
            'unique_photos': 0
        }
        
        # Create output directories
        self.categories = {
            'unique': 'unique_photos',
            'duplicates': 'duplicate_photos'
        }
    
    def print_header(self):
        """Print program header information"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Photo Deduplication Tool ==={Colors.ENDC}")
        print(f"{Colors.BLUE}Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    def print_footer(self):
        """Print program footer statistics"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Processing Complete ==={Colors.ENDC}")
        print(f"{Colors.BLUE}End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
        print("\nDeduplication Statistics:")
        print(f"{Colors.GREEN}Total photos processed: {self.stats['processed_photos']}{Colors.ENDC}")
        print(f"{Colors.GREEN}Unique photos: {self.stats['unique_photos']}{Colors.ENDC}")
        print(f"{Colors.GREEN}Duplicate photos: {self.stats['duplicate_photos']}{Colors.ENDC}")
        print(f"{Colors.GREEN}Duplicate groups: {self.stats['duplicate_groups']}{Colors.ENDC}")
    
    def create_directories(self, base_dir):
        """Create output directories"""
        for category in self.categories.values():
            os.makedirs(os.path.join(base_dir, category), exist_ok=True)
    
    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def calculate_image_hash(self, img):
        """Calculate perceptual hash of image"""
        if img is None:
            return None
        try:
            # Convert OpenCV image to PIL Image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            return str(imagehash.average_hash(pil_img))
        except Exception as e:
            logger.error(f"❌ Error calculating image hash: {str(e)}")
            return None
    
    def extract_features(self, img):
        """Extract SIFT features from image"""
        if img is None:
            return None, None
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            return keypoints, descriptors
        except Exception as e:
            logger.error(f"❌ Error extracting features: {str(e)}")
            return None, None
    
    def calculate_similarity(self, desc1, desc2):
        """Calculate similarity between two images using feature matching"""
        if desc1 is None or desc2 is None:
            return 0.0
            
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.feature_threshold * n.distance:
                good_matches.append(m)
        
        return len(good_matches) / min(len(desc1), len(desc2))
    
    def get_image_quality(self, img_path):
        """Get image quality score based on size and resolution"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return 0
            height, width = img.shape[:2]
            file_size = os.path.getsize(img_path)
            return height * width * (file_size / (1024 * 1024))  # Resolution * file size in MB
        except Exception:
            return 0
    
    def find_duplicates(self, image_files):
        """Find duplicate and similar photos"""
        # First pass: Find exact duplicates using file hash
        hash_groups = defaultdict(list)
        for img_path in image_files:
            file_hash = self.calculate_file_hash(img_path)
            hash_groups[file_hash].append(img_path)
        
        # Second pass: Find similar photos using image hash and features
        similar_groups = []
        processed = set()
        
        for img_path in image_files:
            if img_path in processed:
                continue
                
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Calculate image hash
            img_hash = self.calculate_image_hash(img)
            if img_hash is None:
                continue
            
            # Extract features
            keypoints, descriptors = self.extract_features(img)
            if descriptors is None:
                continue
            
            # Find similar images
            similar_group = [img_path]
            processed.add(img_path)
            
            for other_path in image_files:
                if other_path in processed:
                    continue
                
                # Read other image
                other_img = cv2.imread(other_path)
                if other_img is None:
                    continue
                
                # Calculate other image hash
                other_hash = self.calculate_image_hash(other_img)
                if other_hash is None:
                    continue
                
                # Check hash difference
                hash_diff = abs(int(img_hash, 16) - int(other_hash, 16))
                if hash_diff > self.hash_threshold:
                    continue
                
                # Extract other image features
                other_kp, other_desc = self.extract_features(other_img)
                if other_desc is None:
                    continue
                
                # Calculate similarity
                similarity = self.calculate_similarity(descriptors, other_desc)
                if similarity > self.feature_threshold:
                    similar_group.append(other_path)
                    processed.add(other_path)
            
            if len(similar_group) > 1:
                similar_groups.append(similar_group)
        
        return hash_groups, similar_groups
    
    def process_duplicates(self, hash_groups, similar_groups, output_dir, dry_run=False):
        """Process duplicate and similar photos"""
        # Process exact duplicates
        for file_hash, group in hash_groups.items():
            if len(group) > 1:
                self.stats['duplicate_groups'] += 1
                self.stats['duplicate_photos'] += len(group) - 1
                
                # Find best quality image
                best_img = max(group, key=self.get_image_quality)
                
                if dry_run:
                    logger.info(f"\n📋 Duplicate group (File Hash: {file_hash}):")
                    logger.info(f"  Keep: {os.path.basename(best_img)}")
                    for img_path in group:
                        if img_path != best_img:
                            logger.info(f"  Remove: {os.path.basename(img_path)}")
                else:
                    # Copy best image to unique directory
                    shutil.copy2(best_img, os.path.join(output_dir, self.categories['unique'], os.path.basename(best_img)))
                    self.stats['unique_photos'] += 1
                    
                    # Move duplicates to duplicate directory
                    for img_path in group:
                        if img_path != best_img:
                            shutil.copy2(img_path, os.path.join(output_dir, self.categories['duplicates'], os.path.basename(img_path)))
        
        # Process similar photos
        for group in similar_groups:
            self.stats['duplicate_groups'] += 1
            self.stats['duplicate_photos'] += len(group) - 1
            
            # Find best quality image
            best_img = max(group, key=self.get_image_quality)
            
            if dry_run:
                logger.info(f"\n📋 Similar photo group:")
                logger.info(f"  Keep: {os.path.basename(best_img)}")
                for img_path in group:
                    if img_path != best_img:
                        logger.info(f"  Remove: {os.path.basename(img_path)}")
            else:
                # Copy best image to unique directory
                shutil.copy2(best_img, os.path.join(output_dir, self.categories['unique'], os.path.basename(best_img)))
                self.stats['unique_photos'] += 1
                
                # Move duplicates to duplicate directory
                for img_path in group:
                    if img_path != best_img:
                        shutil.copy2(img_path, os.path.join(output_dir, self.categories['duplicates'], os.path.basename(img_path)))
    
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
        
        # Find duplicates
        logger.info(f"{Colors.BLUE}🔍 Finding duplicates...{Colors.ENDC}")
        hash_groups, similar_groups = self.find_duplicates(image_files)
        
        # Process duplicates
        logger.info(f"{Colors.BLUE}📝 Processing duplicates...{Colors.ENDC}")
        self.process_duplicates(hash_groups, similar_groups, output_dir, dry_run)
        
        # Update statistics
        self.stats['processed_photos'] = self.stats['total_photos']

def main():
    parser = argparse.ArgumentParser(description='Photo Deduplication Tool')
    parser.add_argument('input_dir', help='Input directory path')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('--dry-run', action='store_true', help='Preview operations without executing')
    parser.add_argument('--hash-threshold', type=int, default=5, help='Perceptual hash difference threshold')
    parser.add_argument('--feature-threshold', type=float, default=0.7, help='Feature matching threshold')
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"❌ Input directory does not exist: {args.input_dir}")
        return
        
    # Check if input directory is readable
    if not os.access(args.input_dir, os.R_OK):
        logger.error(f"❌ Input directory cannot be accessed (permission issue): {args.input_dir}")
        return
    
    deduplicator = PhotoDeduplicator()
    deduplicator.hash_threshold = args.hash_threshold
    deduplicator.feature_threshold = args.feature_threshold
    deduplicator.print_header()
    deduplicator.process_directory(args.input_dir, args.output_dir, args.dry_run)
    deduplicator.print_footer()

if __name__ == '__main__':
    main() 