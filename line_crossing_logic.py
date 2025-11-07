#!/usr/bin/env python3
"""
line_crossing_logic.py
Virtual Line-Crossing Detection for Directional Analytics
Uses trajectory-based analysis with cross-product mathematics
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Optional, Dict


class DirectionFinder:
    """
    Implements virtual line-crossing detection for directional vehicle analytics.
    
    Uses the cross-product method to determine which side of a line a point is on,
    and tracks trajectory history to detect when vehicles cross the virtual line.
    
    Attributes:
        line_coords: Tuple of two points defining the virtual line [(x1, y1), (x2, y2)]
        trajectory_history: Dict mapping object_id to list of past positions
        crossing_states: Dict mapping object_id to their last known line side
        history_length: Number of frames to keep in trajectory history
    """
    
    def __init__(self, line_coords: list, history_length: int = 30):
        """
        Initialize the DirectionFinder
        
        Args:
            line_coords: Virtual line defined as [(x1, y1), (x2, y2)]
            history_length: Max frames to keep in trajectory buffer
        """
        self.line_coords = line_coords
        self.line_p1 = np.array(line_coords[0], dtype=np.float32)
        self.line_p2 = np.array(line_coords[1], dtype=np.float32)
        
        # Storage for trajectory analysis
        self.trajectory_history: Dict[int, list] = defaultdict(list)
        self.crossing_states: Dict[int, int] = {}  # -1, 0, or 1
        self.crossing_debounce: Dict[int, int] = defaultdict(int)
        
        self.history_length = history_length
        
        # Debounce parameters to avoid duplicate crossings
        self.DEBOUNCE_FRAMES = 10  # Ignore crossings for N frames after detection
        
    
    def _compute_side(self, point: np.ndarray) -> int:
        """
        Compute which side of the line a point is on using cross product.
        
        Cross product formula:
        CP = (x - x1)(y2 - y1) - (y - y1)(x2 - x1)
        
        Returns:
            -1: Point is on the left/below the line
             0: Point is on the line
             1: Point is on the right/above the line
        """
        x, y = point
        x1, y1 = self.line_p1
        x2, y2 = self.line_p2
        
        cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        
        # Use small epsilon for numerical stability
        epsilon = 1e-6
        if abs(cross_product) < epsilon:
            return 0
        return 1 if cross_product > 0 else -1
    
    
    def _detect_crossing(self, object_id: int, current_side: int) -> Optional[str]:
        """
        Detect if a line crossing occurred and determine direction
        
        Args:
            object_id: Unique tracker ID
            current_side: Current side of line (-1 or 1)
            
        Returns:
            "towards_camera": Vehicle crossed from side 1 to -1
            "away_from_camera": Vehicle crossed from side -1 to 1
            None: No crossing detected
        """
        # Check debounce - ignore if recently crossed
        if self.crossing_debounce[object_id] > 0:
            self.crossing_debounce[object_id] -= 1
            return None
        
        # First time seeing this object
        if object_id not in self.crossing_states:
            self.crossing_states[object_id] = current_side
            return None
        
        previous_side = self.crossing_states[object_id]
        
        # Detect crossing: sides must be different and non-zero
        if previous_side != current_side and previous_side != 0 and current_side != 0:
            # Update state
            self.crossing_states[object_id] = current_side
            
            # Set debounce counter
            self.crossing_debounce[object_id] = self.DEBOUNCE_FRAMES
            
            # Determine direction based on line orientation
            # Convention: 1 -> -1 is "towards", -1 -> 1 is "away"
            if previous_side == 1 and current_side == -1:
                return "towards_camera"
            elif previous_side == -1 and current_side == 1:
                return "away_from_camera"
        
        # Update current state even if no crossing
        self.crossing_states[object_id] = current_side
        return None
    
    
    def check_direction(
        self, 
        object_id: int, 
        current_x: float, 
        current_y: float
    ) -> Optional[str]:
        """
        Main API: Check if vehicle crossed the line and return direction
        
        Args:
            object_id: Unique tracking ID from ByteTrack
            current_x: Current centroid X coordinate
            current_y: Current centroid Y coordinate
            
        Returns:
            Direction string ("towards_camera" or "away_from_camera") or None
        """
        current_point = np.array([current_x, current_y], dtype=np.float32)
        
        # Add to trajectory history for analytics
        self.trajectory_history[object_id].append((current_x, current_y))
        
        # Maintain sliding window
        if len(self.trajectory_history[object_id]) > self.history_length:
            self.trajectory_history[object_id].pop(0)
        
        # Compute which side of the line the vehicle is on
        current_side = self._compute_side(current_point)
        
        # Detect crossing
        direction = self._detect_crossing(object_id, current_side)
        
        return direction
    
    
    def get_trajectory(self, object_id: int) -> list:
        """
        Get the full trajectory history for a given object
        
        Args:
            object_id: Tracking ID
            
        Returns:
            List of (x, y) tuples representing past positions
        """
        return self.trajectory_history.get(object_id, [])
    
    
    def cleanup_old_tracks(self, active_ids: set):
        """
        Remove trajectory data for tracks that are no longer active
        
        Args:
            active_ids: Set of currently active object IDs
        """
        # Get all stored IDs
        stored_ids = set(self.trajectory_history.keys())
        
        # Find inactive IDs
        inactive_ids = stored_ids - active_ids
        
        # Remove inactive tracks to prevent memory bloat
        for obj_id in inactive_ids:
            if obj_id in self.trajectory_history:
                del self.trajectory_history[obj_id]
            if obj_id in self.crossing_states:
                del self.crossing_states[obj_id]
            if obj_id in self.crossing_debounce:
                del self.crossing_debounce[obj_id]
    
    
    def reset(self):
        """Reset all tracking state - useful for new video streams"""
        self.trajectory_history.clear()
        self.crossing_states.clear()
        self.crossing_debounce.clear()


# ============================================================================
# ADVANCED: Multi-Line Support (for complex intersections)
# ============================================================================

class MultiLineDirectionFinder:
    """
    Extended version supporting multiple virtual lines
    Useful for roundabouts with multiple entry/exit points
    """
    
    def __init__(self, lines_config: Dict[str, list]):
        """
        Args:
            lines_config: Dict mapping line names to coordinates
            Example: {
                "north_entry": [(100, 200), (300, 200)],
                "south_exit": [(100, 800), (300, 800)]
            }
        """
        self.lines = {
            name: DirectionFinder(coords) 
            for name, coords in lines_config.items()
        }
    
    
    def check_all_lines(
        self, 
        object_id: int, 
        current_x: float, 
        current_y: float
    ) -> Dict[str, Optional[str]]:
        """
        Check vehicle against all virtual lines
        
        Returns:
            Dict mapping line names to direction events
        """
        results = {}
        for line_name, finder in self.lines.items():
            direction = finder.check_direction(object_id, current_x, current_y)
            if direction:
                results[line_name] = direction
        return results
    
    
    def cleanup_old_tracks(self, active_ids: set):
        """Cleanup all line finders"""
        for finder in self.lines.values():
            finder.cleanup_old_tracks(active_ids)


# ============================================================================
# UNIT TEST
# ============================================================================

if __name__ == "__main__":
    """Simple test of line crossing logic"""
    
    print("Testing DirectionFinder...")
    
    # Create vertical line at x=500
    line = [(500, 0), (500, 1000)]
    finder = DirectionFinder(line)
    
    # Simulate vehicle moving from right to left (towards camera)
    trajectory = [
        (600, 500),  # Right side
        (550, 500),  # Still right
        (500, 500),  # On line
        (450, 500),  # Crossed to left
        (400, 500),  # Left side
    ]
    
    print("\nSimulating trajectory: right -> left (towards_camera)")
    for i, (x, y) in enumerate(trajectory):
        result = finder.check_direction(object_id=1, current_x=x, current_y=y)
        print(f"Frame {i}: position=({x}, {y}), direction={result}")
    
    # Test opposite direction
    print("\nSimulating trajectory: left -> right (away_from_camera)")
    finder.reset()
    
    trajectory_reverse = [
        (400, 500),  # Left side
        (450, 500),  # Still left
        (500, 500),  # On line
        (550, 500),  # Crossed to right
        (600, 500),  # Right side
    ]
    
    for i, (x, y) in enumerate(trajectory_reverse):
        result = finder.check_direction(object_id=2, current_x=x, current_y=y)
        print(f"Frame {i}: position=({x}, {y}), direction={result}")
    
    print("\n✓ DirectionFinder test complete")
