"""Prompt strategies for evaluating SAM's robustness to different input types.

Each strategy generates point_coords and point_labels from a binary mask,
simulating different ways a user might prompt SAM. This is central to the
paper's thesis: specialized decoder fine-tuning improves robustness across
ALL prompt types, not just the training prompt type.
"""

import cv2
import numpy as np


class PromptStrategy:
    """Collection of prompting strategies for SAM evaluation."""

    @staticmethod
    def center_of_mass(mask: np.ndarray, num_points: int = 1):
        """Center-of-mass prompting (baseline single-point strategy).

        Args:
            mask: Binary mask (H, W).
            num_points: Ignored (always returns 1 point).

        Returns:
            (point_coords, point_labels) or (None, None) if mask is empty.
        """
        ys, xs = np.where(mask > 128)
        if len(xs) == 0:
            return None, None
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        return np.array([[cx, cy]]), np.array([1])

    @staticmethod
    def edge_points(mask: np.ndarray, num_points: int = 1):
        """Edge-based prompting: sample points along object boundary.

        Uses cv2.findContours to extract the object boundary, then
        samples evenly-spaced points along the largest contour.

        Args:
            mask: Binary mask (H, W).
            num_points: Number of edge points to sample.

        Returns:
            (point_coords, point_labels) or (None, None) if mask is empty.
        """
        ys, xs = np.where(mask > 128)
        if len(xs) == 0:
            return None, None

        mask_uint8 = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return PromptStrategy.center_of_mass(mask, num_points)

        contour = max(contours, key=cv2.contourArea)
        num_points = min(num_points, len(contour))

        indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
        points = [contour[i][0] for i in indices]

        point_coords = np.array(points)
        point_labels = np.ones(num_points, dtype=int)

        return point_coords, point_labels

    @staticmethod
    def multi_point_grid(mask: np.ndarray, num_points: int = 4):
        """Multi-point grid: sample points in a grid over the object bbox.

        Creates a grid inside the object bounding box and keeps only
        points that fall on foreground pixels.

        Args:
            mask: Binary mask (H, W).
            num_points: Target number of points (uses sqrt for grid).

        Returns:
            (point_coords, point_labels) or (None, None) if mask is empty.
        """
        ys, xs = np.where(mask > 128)
        if len(xs) == 0:
            return None, None

        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)

        grid_size = max(1, int(np.sqrt(num_points)))

        x_coords = np.linspace(x_min, x_max, grid_size + 2)[1:-1]
        y_coords = np.linspace(y_min, y_max, grid_size + 2)[1:-1]

        points = []
        for x in x_coords:
            for y in y_coords:
                xi, yi = int(x), int(y)
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                    if mask[yi, xi] > 128:
                        points.append([xi, yi])

        if len(points) == 0:
            return PromptStrategy.center_of_mass(mask, 1)

        return np.array(points), np.ones(len(points), dtype=int)

    @staticmethod
    def random_points(mask: np.ndarray, num_points: int = 3):
        """Random foreground point sampling.

        Randomly selects points from the foreground region of the mask.

        Args:
            mask: Binary mask (H, W).
            num_points: Number of random points to sample.

        Returns:
            (point_coords, point_labels) or (None, None) if mask is empty.
        """
        ys, xs = np.where(mask > 128)
        if len(xs) == 0:
            return None, None

        num_points = min(num_points, len(xs))
        indices = np.random.choice(len(xs), num_points, replace=False)

        point_coords = np.array([[xs[i], ys[i]] for i in indices])
        point_labels = np.ones(num_points, dtype=int)

        return point_coords, point_labels


# Registry mapping strategy names to their configs
PROMPT_CONFIGS = {
    "center": {
        "name": "Center-of-Mass (Single)",
        "strategy": PromptStrategy.center_of_mass,
        "num_points": 1,
        "description": "Single point at object center of mass",
    },
    "edge_single": {
        "name": "Edge (Single)",
        "strategy": PromptStrategy.edge_points,
        "num_points": 1,
        "description": "Single point on object boundary",
    },
    "multi_grid": {
        "name": "Multi-Point Grid (4 pts)",
        "strategy": PromptStrategy.multi_point_grid,
        "num_points": 4,
        "description": "Grid of 4 points covering object",
    },
    "multi_random": {
        "name": "Multi-Point Random (3 pts)",
        "strategy": PromptStrategy.random_points,
        "num_points": 3,
        "description": "Three random foreground points",
    },
}
