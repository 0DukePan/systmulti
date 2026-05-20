"""
Generate synthetic test frames for pipeline demonstration.

Creates 30 frames (320×240) showing:
 - Animated color gradient background (slowly shifting hue)
 - A moving white rectangle (simulating motion between frames)
 - A moving circle (secondary object)
"""

import numpy as np
import cv2
import os

N_FRAMES  = 30
WIDTH     = 320
HEIGHT    = 240
OUT_DIR   = 'frames'


def make_frame(i: int, n: int) -> np.ndarray:
    t = i / max(n - 1, 1)   # 0.0 → 1.0

    # ── Gradient background (hue shifts over time) ──
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            hue = int((x / WIDTH * 120 + t * 60)) % 180
            frame[y, x] = [hue, 200, 180]

    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    # ── Moving rectangle ──
    rx = int(20 + t * (WIDTH - 80))
    ry = int(HEIGHT // 2 - 20)
    cv2.rectangle(frame, (rx, ry), (rx + 60, ry + 40), (255, 255, 255), -1)

    # ── Moving circle ──
    cx = int(WIDTH // 2 + np.sin(t * 2 * np.pi) * (WIDTH // 4))
    cy = int(HEIGHT // 4)
    cv2.circle(frame, (cx, cy), 20, (0, 80, 255), -1)

    # ── Frame number label ──
    cv2.putText(frame, f'Frame {i+1:02d}/{n}',
                (5, HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating {N_FRAMES} frames ({WIDTH}×{HEIGHT}) → '{OUT_DIR}/'")
    for i in range(N_FRAMES):
        frame = make_frame(i, N_FRAMES)
        path = os.path.join(OUT_DIR, f'frame_{i:04d}.png')
        cv2.imwrite(path, frame)
    print(f"Done. {N_FRAMES} frames saved to '{OUT_DIR}/'.")


if __name__ == '__main__':
    main()
