#!/usr/bin/env python3
"""
Auto-annotate captured images with initial bounding boxes.

Uses simple edge detection + contour finding to suggest card regions.
You should then open the images in LabelImg and refine the bboxes.

Usage:
    python auto_annotate.py --input-dir data/collected/front --class-name id-front
    python auto_annotate.py --input-dir data/collected/back --class-name id-back
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET


def find_card_bbox(image_path):
    """Find the largest rectangular contour in the image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur and edge detect
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_bbox = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.05):  # Ignore tiny regions
            continue
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        
        # Score by area and aspect ratio closeness to ID card (~1.58)
        aspect_score = max(0, 1 - abs(aspect - 1.58) / 1.58)
        score = area * aspect_score
        
        if score > best_score:
            best_score = score
            best_bbox = (x, y, x + cw, y + ch)
    
    # Fallback: center crop if no good contour found
    if best_bbox is None:
        margin = int(min(w, h) * 0.1)
        best_bbox = (margin, margin, w - margin, h - margin)
    
    return best_bbox


def create_xml(image_path, bbox, class_name, output_path):
    """Create a LabelImg-compatible XML file."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = image_path.parent.name
    ET.SubElement(annotation, "filename").text = image_path.name
    ET.SubElement(annotation, "path").text = str(image_path)
    
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"
    
    ET.SubElement(annotation, "segmented").text = "0"
    
    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = class_name
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(x1)
    ET.SubElement(bndbox, "ymin").text = str(y1)
    ET.SubElement(bndbox, "xmax").text = str(x2)
    ET.SubElement(bndbox, "ymax").text = str(y2)
    
    tree = ET.ElementTree(annotation)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate images with initial bboxes")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory with images")
    parser.add_argument("--class-name", "-c", required=True, help="Class name for annotations")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg")) + sorted(input_dir.glob("*.png"))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Auto-annotating with class '{args.class_name}'...")
    
    success = 0
    fallback = 0
    
    for img_path in image_files:
        bbox = find_card_bbox(img_path)
        if bbox is None:
            continue
        
        xml_path = output_dir / img_path.with_suffix(".xml").name
        create_xml(img_path, bbox, args.class_name, xml_path)
        success += 1
        
        # Check if it was a fallback center crop
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        margin = int(min(w, h) * 0.1)
        if bbox == (margin, margin, w - margin, h - margin):
            fallback += 1
    
    print(f"\n✅ Created {success} annotations")
    print(f"   {success - fallback} with detected contours")
    print(f"   {fallback} with fallback center crop (need manual fix)")
    print(f"\nNext step: Open {output_dir} in LabelImg and refine the bboxes.")


if __name__ == "__main__":
    main()
