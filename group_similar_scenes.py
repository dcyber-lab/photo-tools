#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
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

class SceneGrouper:
    def __init__(self):
        # Initialize SIFT feature detector
        self.sift = cv2.SIFT_create()
        
        # Initialize FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Set similarity threshold
        self.similarity_threshold = 0.7
        
        # Statistics
        self.total_images = 0
        self.processed_images = 0
        self.scene_count = 0
    
    def print_header(self):
        """Print program header information"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Scene Similarity Grouping Tool ==={Colors.ENDC}")
        print(f"{Colors.BLUE}Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    def print_footer(self):
        """Print program footer statistics"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== Processing Complete ==={Colors.ENDC}")
        print(f"{Colors.BLUE}End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
        print(f"\n{Colors.GREEN}Total images processed: {self.processed_images}{Colors.ENDC}")
        print(f"{Colors.GREEN}Total scenes identified: {self.scene_count}{Colors.ENDC}")
    
    def extract_features(self, img):
        """Extract SIFT features from image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def calculate_similarity(self, desc1, desc2):
        """Calculate similarity between two images using feature matching"""
        if desc1 is None or desc2 is None:
            return 0.0
            
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.similarity_threshold * n.distance:
                good_matches.append(m)
        
        return len(good_matches) / min(len(desc1), len(desc2))
    
    def find_similar_scenes(self, images, features):
        """Find similar scenes and group them"""
        scenes = []
        processed = set()
        
        for i, (img_path, (kp1, desc1)) in enumerate(images):
            if img_path in processed:
                continue
                
            current_scene = [img_path]
            processed.add(img_path)
            
            for j, (other_path, (kp2, desc2)) in enumerate(images[i+1:], i+1):
                if other_path in processed:
                    continue
                    
                similarity = self.calculate_similarity(desc1, desc2)
                if similarity > self.similarity_threshold:
                    current_scene.append(other_path)
                    processed.add(other_path)
            
            if len(current_scene) > 1:
                scenes.append(current_scene)
        
        return scenes
    
    def rename_files(self, scenes, output_dir, dry_run=False):
        """Rename files according to scene groups"""
        for scene_idx, scene in enumerate(scenes, 1):
            for img_idx, img_path in enumerate(scene, 1):
                # Get file extension
                ext = Path(img_path).suffix.lower()
                
                # Create new filename
                new_name = f"scene_{scene_idx:03d}_{img_idx:03d}{ext}"
                target_path = os.path.join(output_dir, new_name)
                
                if dry_run:
                    logger.info(f"📋 Preview: {os.path.basename(img_path)} -> {new_name}")
                else:
                    try:
                        os.rename(img_path, target_path)
                        logger.info(f"✅ Renamed: {os.path.basename(img_path)} -> {new_name}")
                    except Exception as e:
                        logger.error(f"❌ Rename failed: {os.path.basename(img_path)}: {str(e)}")
    
    def process_directory(self, input_dir, output_dir, dry_run=False):
        """Process all images in directory"""
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.arw', '.cr2', '.nef'}
        
        # Get all image files
        images = []
        features = []
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    img_path = os.path.join(root, file)
                    images.append(img_path)
        
        self.total_images = len(images)
        if self.total_images == 0:
            logger.warning(f"{Colors.YELLOW}⚠️ No image files found{Colors.ENDC}")
            return
        
        logger.info(f"{Colors.BLUE}📁 Found {self.total_images} image files{Colors.ENDC}")
        
        # Extract features from all images
        for i, img_path in enumerate(images, 1):
            # Show progress
            progress = (i / self.total_images) * 100
            sys.stdout.write(f"\r{Colors.BLUE}Extracting features: {progress:.1f}% ({i}/{self.total_images}){Colors.ENDC}")
            sys.stdout.flush()
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.error(f"❌ Cannot read image: {img_path}")
                    continue
                
                kp, desc = self.extract_features(img)
                features.append((kp, desc))
                self.processed_images += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {img_path}: {str(e)}")
        
        print("\n")  # New line to avoid progress bar overlap
        
        # Find similar scenes
        logger.info(f"{Colors.BLUE}🔍 Finding similar scenes...{Colors.ENDC}")
        scenes = self.find_similar_scenes(images, features)
        self.scene_count = len(scenes)
        
        if self.scene_count == 0:
            logger.warning(f"{Colors.YELLOW}⚠️ No similar scenes found{Colors.ENDC}")
            return
        
        logger.info(f"{Colors.GREEN}✨ Found {self.scene_count} scenes{Colors.ENDC}")
        
        # Rename files
        logger.info(f"{Colors.BLUE}📝 Renaming files...{Colors.ENDC}")
        self.rename_files(scenes, output_dir, dry_run)

def main():
    parser = argparse.ArgumentParser(description='Scene Similarity Grouping Tool')
    parser.add_argument('input_dir', help='Input directory path')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('--dry-run', action='store_true', help='Preview operations without executing')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold (0.0-1.0)')
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"❌ Input directory does not exist: {args.input_dir}")
        return
        
    # Check if input directory is readable
    if not os.access(args.input_dir, os.R_OK):
        logger.error(f"❌ Input directory cannot be accessed (permission issue): {args.input_dir}")
        return
    
    grouper = SceneGrouper()
    grouper.similarity_threshold = args.threshold
    grouper.print_header()
    grouper.process_directory(args.input_dir, args.output_dir, args.dry_run)
    grouper.print_footer()

if __name__ == '__main__':
    main() 