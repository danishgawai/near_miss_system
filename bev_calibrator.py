#!/usr/bin/env python3
"""
BEV (Bird's-Eye View) calibration tool for video/stream.

- Grabs a frame from the video, shows it on the LEFT.
- Shows a gridded BEV canvas on the RIGHT.
- Click ≥4 matched point pairs (src on left, dst on right).
- Computes homography (RANSAC if >4 points).
- Shows BEV preview with verification overlays.
- Runs scale calibration to compute pixels_per_meter.
- Saves bev_config.json with homography + pixels_per_meter.

Controls:
  Left-click:  Select points (alternating: left=src, right=dst)
  'r':         Reset all points
  ENTER:       Compute homography and show preview (need ≥4 pairs)
  In preview — 's': save config, 'q': quit without saving
  'q' in main window: quit without computing
"""

import argparse
import json
import os
from typing import List, Tuple

import cv2
import numpy as np


WINDOW_NAME = "BEV Calibration (Left: Real, Right: BEV)"
BEV_PREVIEW_WINDOW = "BEV Preview"
SCALE_WINDOW = "Scale Calibration"

# Colors for point pairs (BGR)
PAIR_COLORS = [
    (0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 200),
    (200, 0, 200), (200, 200, 0), (128, 0, 255), (255, 128, 0),
]


class VideoBEVCalibrator:
    def __init__(
        self,
        video_source: str,
        frame_index: int = 0,
        output_config: str = "bev_config.json",
    ):
        self.video_source = video_source
        self.frame_index = max(0, int(frame_index))
        self.output_config = output_config

        # Extract a calibration frame
        self.frame = self._get_frame_from_video()
        if self.frame is None:
            raise RuntimeError(
                f"Failed to get frame {self.frame_index} from '{video_source}'"
            )

        self.h, self.w = self.frame.shape[:2]

        # Build combined canvas
        self._build_canvas()

        # Points in image coordinates
        self.src_points: List[Tuple[int, int]] = []
        self.dst_points: List[Tuple[int, int]] = []

        self.homography = None
        self.pixels_per_meter = 100.0  # Default; overridden by scale calibration

    def _get_frame_from_video(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.video_source}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0 and self.frame_index >= total:
            print(
                f"[WARN] frame_index {self.frame_index} >= total frames {total}. "
                "Using last frame."
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)

        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _build_canvas(self):
        """Create the side-by-side canvas with grid on the BEV side."""
        self.combined = np.zeros((self.h, self.w * 2, 3), dtype=np.uint8)
        self.combined[:, : self.w] = self.frame
        self.combined[:, self.w :] = 255  # white BEV canvas

        # Grid on right side
        grid_spacing = 50
        grid_color = (210, 210, 210)
        for gx in range(0, self.w, grid_spacing):
            cv2.line(
                self.combined, (self.w + gx, 0), (self.w + gx, self.h), grid_color, 1
            )
        for gy in range(0, self.h, grid_spacing):
            cv2.line(
                self.combined, (self.w, gy), (self.w * 2, gy), grid_color, 1
            )

        self._draw_instructions()

        # Keep a clean copy for resets
        self._base_combined = self.combined.copy()

    def _draw_instructions(self):
        cv2.putText(
            self.combined, "REAL FRAME", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.putText(
            self.combined, "BEV CANVAS", (self.w + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        hints = [
            "1) Click points on LEFT (src), then matching point on RIGHT (dst).",
            "2) Repeat for >=4 pairs. More pairs = more robust calibration.",
            "3) Press ENTER to compute.  'r' = reset.  'q' = quit.",
        ]
        for i, line in enumerate(hints):
            cv2.putText(
                self.combined, line, (20, 50 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
            )

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if x < self.w:
            # --- Left side: SRC point ---
            # Only accept if we don't have an unmatched src already
            if len(self.src_points) > len(self.dst_points):
                print("[WARN] Click the corresponding DST point on the right first.")
                return

            self.src_points.append((x, y))
            idx = len(self.src_points)
            color = PAIR_COLORS[(idx - 1) % len(PAIR_COLORS)]
            print(f"[SRC {idx}] ({x}, {y})")

            cv2.circle(self.combined, (x, y), 5, color, -1)
            cv2.putText(
                self.combined, f"S{idx}", (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
        else:
            # --- Right side: DST point ---
            if len(self.dst_points) >= len(self.src_points):
                print("[WARN] Click a SRC point on the left first.")
                return

            x_local = x - self.w
            self.dst_points.append((x_local, y))
            idx = len(self.dst_points)
            color = PAIR_COLORS[(idx - 1) % len(PAIR_COLORS)]
            print(f"[DST {idx}] ({x_local}, {y})")

            cv2.circle(self.combined, (x, y), 5, color, -1)
            cv2.putText(
                self.combined, f"D{idx}", (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

            # Draw connecting line between matched src and dst
            src_pt = self.src_points[idx - 1]
            cv2.line(self.combined, src_pt, (x, y), color, 1, cv2.LINE_AA)

    def collect_points(self) -> bool:
        print(
            "\n[INSTRUCTIONS]\n"
            "1) Click a point on LEFT, then the matching point on RIGHT.\n"
            "2) Repeat for ≥4 pairs (more = better accuracy with RANSAC).\n"
            "3) Press ENTER when done.  'r' to reset.  'q' to quit.\n"
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            cv2.imshow(WINDOW_NAME, self.combined)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                print("[INFO] Quitting without computing homography.")
                cv2.destroyWindow(WINDOW_NAME)
                return False

            if key == ord("r"):
                print("[INFO] Resetting all points.")
                self.src_points.clear()
                self.dst_points.clear()
                self.combined = self._base_combined.copy()

            if key in (13, 10):  # ENTER
                n_src, n_dst = len(self.src_points), len(self.dst_points)
                if n_src >= 4 and n_src == n_dst:
                    print(
                        f"[INFO] {n_src} matched pairs collected. Computing homography."
                    )
                    break
                else:
                    print(
                        f"[WARN] Need ≥4 matched pairs. "
                        f"Currently: {n_src} src, {n_dst} dst."
                    )

        cv2.destroyWindow(WINDOW_NAME)
        return True

    @staticmethod
    def _quad_area(points):
        """Shoelace area for a polygon. Used to detect degenerate shapes."""
        pts = np.array(points, dtype=np.float64)
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def compute_homography(self) -> bool:
        n = len(self.src_points)
        if n < 4 or n != len(self.dst_points):
            print("[ERROR] Need ≥4 matched point pairs.")
            return False

        # Validate that points aren't degenerate (for exactly 4 points)
        if n == 4:
            for label, pts in [("SRC", self.src_points), ("DST", self.dst_points)]:
                area = self._quad_area(pts)
                if area < 100:
                    print(
                        f"[ERROR] {label} points form a degenerate shape "
                        f"(area={area:.1f} px²). Press 'r' to reset."
                    )
                    return False

        src = np.array(self.src_points, dtype=np.float32)
        dst = np.array(self.dst_points, dtype=np.float32)

        if n == 4:
            H = cv2.getPerspectiveTransform(src, dst)
        else:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                print("[ERROR] findHomography failed — points may be degenerate.")
                return False
            inliers = int(mask.ravel().sum())
            print(f"[INFO] RANSAC: {inliers}/{n} inliers.")

        self.homography = H
        print("\n[INFO] Homography matrix H (3×3):")
        for row in H:
            print(f"  [{row[0]:12.6f}  {row[1]:12.6f}  {row[2]:12.6f}]")

        # Compute reprojection error
        src_h = np.hstack([src, np.ones((n, 1), dtype=np.float32)])
        projected = (H @ src_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        errors = np.linalg.norm(projected - dst, axis=1)
        print(f"[INFO] Reprojection error: mean={errors.mean():.2f}px, max={errors.max():.2f}px")

        return True

    def show_bev_preview(self) -> str:
        """Show BEV warped image with verification overlays. Returns 'save' or 'quit'."""
        if self.homography is None:
            raise RuntimeError("Homography not computed.")

        bev = cv2.warpPerspective(self.frame, self.homography, (self.w, self.h))

        # Overlay DST points (green circles)
        for i, (x, y) in enumerate(self.dst_points):
            cv2.circle(bev, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(
                bev, f"D{i+1}", (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        # Overlay warped SRC points (red crosses — should overlap green if accurate)
        for i, (sx, sy) in enumerate(self.src_points):
            pt = np.array([[[sx, sy]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(pt, self.homography)
            wx, wy = int(warped[0][0][0]), int(warped[0][0][1])
            cv2.drawMarker(bev, (wx, wy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

        cv2.putText(
            bev,
            "Green=DST, Red X=Warped SRC (should overlap). 's'=save, 'q'=quit",
            (10, self.h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )

        cv2.namedWindow(BEV_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(BEV_PREVIEW_WINDOW, bev)

        print(
            "\n[BEV PREVIEW]\n"
            "Verify: red crosses should overlap green circles.\n"
            "Press 's' to proceed to scale calibration + save.\n"
            "Press 'q' to quit without saving.\n"
        )

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                cv2.destroyWindow(BEV_PREVIEW_WINDOW)
                return "save"
            elif key == ord("q"):
                cv2.destroyWindow(BEV_PREVIEW_WINDOW)
                return "quit"

    def calibrate_scale(self):
        """
        Let the user click 2 points on the BEV image and enter the known
        real-world distance to compute pixels_per_meter.
        """
        if self.homography is None:
            raise RuntimeError("Homography not computed yet.")

        bev = cv2.warpPerspective(self.frame, self.homography, (self.w, self.h))
        scale_points: List[Tuple[int, int]] = []

        def scale_mouse_cb(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN or len(scale_points) >= 2:
                return
            scale_points.append((x, y))
            idx = len(scale_points)
            cv2.circle(bev, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                bev, f"P{idx}", (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            if len(scale_points) == 2:
                cv2.line(bev, scale_points[0], scale_points[1], (0, 255, 0), 2)
                px_dist = np.linalg.norm(
                    np.array(scale_points[0], np.float64) -
                    np.array(scale_points[1], np.float64)
                )
                cv2.putText(
                    bev, f"{px_dist:.1f} px", 
                    ((scale_points[0][0] + scale_points[1][0]) // 2,
                     (scale_points[0][1] + scale_points[1][1]) // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
                )
            cv2.imshow(SCALE_WINDOW, bev)

        cv2.namedWindow(SCALE_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(SCALE_WINDOW, scale_mouse_cb)

        cv2.putText(
            bev,
            "Click 2 points with KNOWN real-world distance, then ENTER. 'q' to skip.",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1,
        )

        print(
            "\n[SCALE CALIBRATION]\n"
            "Click 2 points on the BEV image between which you know\n"
            "the real-world distance (e.g. a lane width, car length).\n"
            "Press ENTER when both points are placed. 'q' to skip.\n"
        )

        while True:
            cv2.imshow(SCALE_WINDOW, bev)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                print(
                    f"[INFO] Scale calibration skipped. "
                    f"Using default pixels_per_meter={self.pixels_per_meter}"
                )
                cv2.destroyWindow(SCALE_WINDOW)
                return

            if key in (13, 10) and len(scale_points) == 2:
                break

        cv2.destroyWindow(SCALE_WINDOW)

        pixel_dist = np.linalg.norm(
            np.array(scale_points[0], dtype=np.float64) -
            np.array(scale_points[1], dtype=np.float64)
        )

        # Get real-world distance from user
        while True:
            real_dist_str = input(
                "Enter the real-world distance between the 2 points (meters): "
            ).strip()
            try:
                real_dist = float(real_dist_str)
                if real_dist <= 0:
                    raise ValueError("Must be positive")
                break
            except ValueError:
                print("[WARN] Enter a positive number. Try again.")

        self.pixels_per_meter = pixel_dist / real_dist
        print(
            f"[INFO] pixels_per_meter = {self.pixels_per_meter:.2f} "
            f"({pixel_dist:.1f} px / {real_dist:.2f} m)"
        )

    def save_config(self):
        if self.homography is None:
            raise RuntimeError("Homography not computed.")

        config_dir = os.path.dirname(self.output_config)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)

        config = {
            "video_source": self.video_source,
            "frame_index": int(self.frame_index),
            "frame_width": int(self.w),
            "frame_height": int(self.h),
            "bev_width_px": int(self.w),
            "bev_height_px": int(self.h),
            "src_points": [[int(x), int(y)] for (x, y) in self.src_points],
            "dst_points": [[int(x), int(y)] for (x, y) in self.dst_points],
            "homography_matrix": self.homography.astype(float).tolist(),
            "pixels_per_meter": round(self.pixels_per_meter, 4),
        }

        try:
            with open(self.output_config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print(f"[INFO] Config saved to: {os.path.abspath(self.output_config)}")
        except IOError as e:
            print(f"[ERROR] Failed to save config: {e}")

    def run(self):
        """Main calibration workflow."""
        if not self.collect_points():
            return

        if not self.compute_homography():
            print("[INFO] Homography computation failed. Exiting.")
            return

        choice = self.show_bev_preview()
        if choice == "quit":
            print("[INFO] Exiting without saving.")
            return

        self.calibrate_scale()
        self.save_config()

        print("\n[DONE] Calibration complete. Config is ready for near-miss detection.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "BEV calibration tool: select ≥4 point correspondences between "
            "a real video frame and a BEV canvas, compute homography, "
            "calibrate scale (pixels_per_meter), and save config."
        )
    )
    parser.add_argument(
        "--video", required=True,
        help="Path/URL to video file or RTSP/HTTP stream.",
    )
    parser.add_argument(
        "--frame-index", type=int, default=0,
        help="Frame index to use for calibration (default: 0).",
    )
    parser.add_argument(
        "--output-config", default="bev_config.json",
        help="Output JSON config file path (default: bev_config.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    calibrator = VideoBEVCalibrator(
        video_source=args.video,
        frame_index=args.frame_index,
        output_config=args.output_config,
    )
    calibrator.run()


if __name__ == "__main__":
    main()