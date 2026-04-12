"""Copy-move forgery detection for ID card verification."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CopyMoveMatch:
    """Copy-move match result."""
    source_bbox: Tuple[int, int, int, int]
    target_bbox: Tuple[int, int, int, int]
    similarity: float
    block_index: int


class CopyMoveDetector:
    """
    Detector for copy-move forgery in images.
    
    Copy-move forgery is when a region of an image is copied and pasted
    elsewhere in the same image. This detector uses block matching to
    find similar regions.
    
    Algorithm:
    1. Divide image into overlapping blocks
    2. Compute feature vector for each block (DCT or pixel values)
    3. Sort and compare blocks to find matches
    4. Filter matches by spatial distance and similarity
    """
    
    def __init__(
        self,
        block_size: int = 8,
        overlap: int = 4,
        similarity_threshold: float = 0.9,
        min_distance: int = 50,
        use_dct: bool = True,
    ):
        """
        Initialize copy-move detector.
        
        Args:
            block_size: Size of blocks to compare
            overlap: Overlap between blocks
            similarity_threshold: Minimum similarity for match
            min_distance: Minimum distance between matching blocks
            use_dct: Use DCT features instead of raw pixels
        """
        self.block_size = block_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.min_distance = min_distance
        self.use_dct = use_dct
    
    def detect(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Detect copy-move forgery.
        
        Args:
            image: Input image
            roi: Region of interest (x1, y1, x2, y2)
            
        Returns:
            Detection results
        """
        # Extract ROI if specified
        if roi is not None:
            x1, y1, x2, y2 = roi
            img = image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            img = image
            offset_x, offset_y = 0, 0
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Extract blocks
        blocks, positions = self._extract_blocks(gray)
        
        if len(blocks) < 2:
            return {
                'forgery_detected': False,
                'matches': [],
                'clusters': [],
                'confidence': 0.0
            }
        
        # Find matches
        matches = self._find_matches(blocks, positions)
        
        # Cluster matches
        clusters = self._cluster_matches(matches)
        
        # Calculate confidence
        confidence = self._calculate_confidence(matches, clusters)
        
        # Adjust coordinates for ROI offset
        adjusted_matches = []
        for match in matches:
            adjusted_match = CopyMoveMatch(
                source_bbox=(
                    match.source_bbox[0] + offset_x,
                    match.source_bbox[1] + offset_y,
                    match.source_bbox[2] + offset_x,
                    match.source_bbox[3] + offset_y
                ),
                target_bbox=(
                    match.target_bbox[0] + offset_x,
                    match.target_bbox[1] + offset_y,
                    match.target_bbox[2] + offset_x,
                    match.target_bbox[3] + offset_y
                ),
                similarity=match.similarity,
                block_index=match.block_index
            )
            adjusted_matches.append(adjusted_match)
        
        # Determine if forgery detected
        forgery_detected = len(adjusted_matches) > 3 and confidence > 0.5
        
        return {
            'forgery_detected': forgery_detected,
            'matches': [
                {
                    'source': m.source_bbox,
                    'target': m.target_bbox,
                    'similarity': m.similarity
                }
                for m in adjusted_matches
            ],
            'clusters': clusters,
            'confidence': confidence,
            'num_blocks': len(blocks),
            'num_matches': len(adjusted_matches)
        }
    
    def _extract_blocks(
        self,
        gray: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract overlapping blocks from image.
        
        Returns:
            Tuple of (blocks array, positions list)
        """
        h, w = gray.shape
        blocks = []
        positions = []
        
        step = self.block_size - self.overlap
        
        for y in range(0, h - self.block_size + 1, step):
            for x in range(0, w - self.block_size + 1, step):
                block = gray[y:y+self.block_size, x:x+self.block_size]
                
                # Compute features
                if self.use_dct:
                    features = self._compute_dct_features(block)
                else:
                    features = self._compute_pixel_features(block)
                
                blocks.append(features)
                positions.append((x, y))
        
        return np.array(blocks), positions
    
    def _compute_dct_features(self, block: np.ndarray) -> np.ndarray:
        """Compute DCT features for block."""
        # Apply DCT
        dct = cv2.dct(block.astype(np.float32))
        
        # Take top-left coefficients (low frequencies)
        features = dct[:4, :4].flatten()
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _compute_pixel_features(self, block: np.ndarray) -> np.ndarray:
        """Compute normalized pixel features for block."""
        features = block.flatten().astype(np.float32)
        
        # Normalize
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            features = (features - mean) / std
        
        return features
    
    def _find_matches(
        self,
        blocks: np.ndarray,
        positions: List[Tuple[int, int]]
    ) -> List[CopyMoveMatch]:
        """Find matching blocks."""
        matches = []
        n_blocks = len(blocks)
        
        # For efficiency, use lexicographic sorting
        # Sort blocks and compare neighbors
        indices = np.arange(n_blocks)
        
        # Sort by block features
        sorted_indices = sorted(indices, key=lambda i: tuple(blocks[i]))
        
        # Compare neighbors
        for i in range(len(sorted_indices) - 1):
            idx1 = sorted_indices[i]
            idx2 = sorted_indices[i + 1]
            
            # Calculate similarity
            similarity = np.dot(blocks[idx1], blocks[idx2])
            
            if similarity >= self.similarity_threshold:
                pos1 = positions[idx1]
                pos2 = positions[idx2]
                
                # Check spatial distance
                distance = np.sqrt(
                    (pos1[0] - pos2[0])**2 +
                    (pos1[1] - pos2[1])**2
                )
                
                if distance >= self.min_distance:
                    match = CopyMoveMatch(
                        source_bbox=(
                            pos1[0],
                            pos1[1],
                            pos1[0] + self.block_size,
                            pos1[1] + self.block_size
                        ),
                        target_bbox=(
                            pos2[0],
                            pos2[1],
                            pos2[0] + self.block_size,
                            pos2[1] + self.block_size
                        ),
                        similarity=float(similarity),
                        block_index=idx1
                    )
                    matches.append(match)
        
        return matches
    
    def _cluster_matches(self, matches: List[CopyMoveMatch]) -> List[Dict]:
        """
        Cluster matches into regions.
        
        Groups nearby matches to identify copied regions.
        """
        if not matches:
            return []
        
        # Group by spatial proximity
        clusters = []
        used = set()
        
        for i, match in enumerate(matches):
            if i in used:
                continue
            
            # Start new cluster
            cluster = [match]
            used.add(i)
            
            # Find nearby matches
            for j, other in enumerate(matches[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if close to any match in cluster
                for cm in cluster:
                    dist = self._bbox_distance(cm.source_bbox, other.source_bbox)
                    if dist < self.block_size * 2:
                        cluster.append(other)
                        used.add(j)
                        break
            
            if len(cluster) >= 3:  # Minimum cluster size
                clusters.append({
                    'size': len(cluster),
                    'avg_similarity': np.mean([m.similarity for m in cluster]),
                    'source_region': self._combine_bboxes([m.source_bbox for m in cluster]),
                    'target_region': self._combine_bboxes([m.target_bbox for m in cluster])
                })
        
        return clusters
    
    def _bbox_distance(
        self,
        bbox1: Tuple[int, ...],
        bbox2: Tuple[int, ...]
    ) -> float:
        """Calculate distance between bounding boxes."""
        c1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        c2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    
    def _combine_bboxes(
        self,
        bboxes: List[Tuple[int, ...]]
    ) -> Tuple[int, int, int, int]:
        """Combine multiple bboxes into one."""
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        
        return (x1, y1, x2, y2)
    
    def _calculate_confidence(
        self,
        matches: List[CopyMoveMatch],
        clusters: List[Dict]
    ) -> float:
        """Calculate forgery confidence."""
        if not matches:
            return 0.0
        
        # Factors
        num_matches = len(matches)
        num_clusters = len(clusters)
        avg_similarity = np.mean([m.similarity for m in matches])
        
        # Score components
        match_score = min(1.0, num_matches / 20)  # Normalize to 20 matches
        cluster_score = min(1.0, num_clusters / 3)  # Normalize to 3 clusters
        similarity_score = avg_similarity
        
        # Weighted combination
        confidence = (
            match_score * 0.3 +
            cluster_score * 0.4 +
            similarity_score * 0.3
        )
        
        return float(confidence)
    
    def visualize_result(
        self,
        image: np.ndarray,
        result: Dict,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize copy-move detection results.
        
        Args:
            image: Original image
            result: Detection result
            save_path: Optional path to save
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Color map for matches
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
        ]
        
        # Draw matches
        matches = result.get('matches', [])
        for i, match in enumerate(matches[:20]):  # Limit to 20 for clarity
            color = colors[i % len(colors)]
            
            # Source
            sx1, sy1, sx2, sy2 = match['source']
            cv2.rectangle(vis_image, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(
                vis_image,
                f"S{i+1}",
                (sx1, sy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            
            # Target
            tx1, ty1, tx2, ty2 = match['target']
            cv2.rectangle(vis_image, (tx1, ty1), (tx2, ty2), color, 2)
            cv2.putText(
                vis_image,
                f"T{i+1}",
                (tx1, ty1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            
            # Draw connection line
            sc = ((sx1 + sx2) // 2, (sy1 + sy2) // 2)
            tc = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
            cv2.line(vis_image, sc, tc, color, 1)
        
        # Add status
        status = "FORGERY DETECTED" if result.get('forgery_detected', False) else "NO FORGERY"
        status_color = (0, 0, 255) if result.get('forgery_detected', False) else (0, 255, 0)
        confidence = result.get('confidence', 0.0)
        
        cv2.putText(
            vis_image,
            f"{status} (conf: {confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )
        
        # Add stats
        num_matches = result.get('num_matches', 0)
        num_clusters = len(result.get('clusters', []))
        
        cv2.putText(
            vis_image,
            f"Matches: {num_matches}, Clusters: {num_clusters}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        if save_path is not None:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Test copy-move detector
    print("Copy-Move Detector Test")
    print("=" * 50)
    
    detector = CopyMoveDetector()
    print("Detector initialized!")
    
    # Create test image with copy-move forgery
    cv2 = __import__('cv2')
    image = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Add original pattern
    pattern = np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)
    image[50:100, 50:100] = pattern
    
    # Copy to another location (forgery)
    image[200:250, 300:350] = pattern
    
    # Add another pattern
    pattern2 = np.random.randint(150, 250, (40, 40, 3), dtype=np.uint8)
    image[150:190, 400:440] = pattern2
    image[280:320, 100:140] = pattern2
    
    # Detect
    result = detector.detect(image)
    
    print(f"\nDetection Results:")
    print(f"  Forgery Detected: {result['forgery_detected']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Num Blocks: {result['num_blocks']}")
    print(f"  Num Matches: {result['num_matches']}")
    print(f"  Num Clusters: {len(result['clusters'])}")
    
    for i, cluster in enumerate(result['clusters']):
        print(f"  Cluster {i+1}: {cluster['size']} matches")
    
    print("\nTest completed!")
