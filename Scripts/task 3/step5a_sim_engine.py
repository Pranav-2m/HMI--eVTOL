import cv2
import numpy as np
import threading
import time
import random
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TelemetryFrame:
    # Flight data
    altitude:    float = 80.0       # metres AGL
    speed:       float = 0.0        # km/h
    heading:     float = 0.0        # degrees (0=North)
    battery:     float = 100.0      # percentage
    wind_speed:  float = 8.0        # knots
    wind_dir:    float = 270.0      # degrees

    # GPS
    latitude:    float = 12.9716    # default: Bangalore area
    longitude:   float = 77.5946
    gps_lock:    bool  = True
    gps_quality: str   = "LOCK"     # LOCK / WEAK / LOST

    # Mission
    mission_time:  int   = 0        # seconds elapsed
    mission_status:str   = "ACTIVE" # ACTIVE / RTB / HOLD / EMERGENCY
    lz_safe:       bool  = True     # landing zone safety flag
    lz_score:      float = 0.85     # 0.0–1.0

    # Images (raw numpy arrays)
    rgb_frame:     Optional[np.ndarray] = field(default=None, repr=False)
    thermal_frame: Optional[np.ndarray] = field(default=None, repr=False)
    fused_frame:   Optional[np.ndarray] = field(default=None, repr=False)

    # Frame metadata
    frame_id:    int = 0
    timestamp:   float = 0.0


# ── Telemetry Simulator ────────────────────────────────────
# Generates realistic-looking changes to telemetry values
# each frame. Uses sine waves and random walks to mimic
# real flight behaviour.

class TelemetrySimulator:
    def __init__(self):
        self.frame_id    = 0
        self.start_time  = time.time()
        self.battery     = 95.0
        self.altitude    = 80.0
        self.speed       = 42.0
        self.heading     = 45.0
        self.lat         = 12.9716
        self.lon         = 77.5946
        self.wind_speed  = 8.0
        self.wind_dir    = 270.0
        self.lz_score    = 0.85

    def step(self):
        """Advance telemetry by one frame."""
        t = time.time() - self.start_time

        # Battery drains slowly (~1% per 30 seconds in sim)
        self.battery = max(0, self.battery - 0.008)

        # Altitude oscillates slightly (wind, hover corrections)
        self.altitude = 80.0 + math.sin(t * 0.3) * 4.0 \
                             + random.gauss(0, 0.3)

        # Speed varies as drone moves / hovers
        self.speed = max(0, 42.0 + math.sin(t * 0.15) * 12.0
                            + random.gauss(0, 1.0))

        # Heading drifts slowly
        self.heading = (self.heading + random.gauss(0, 0.3)) % 360

        # GPS position moves slowly (drone traversing area)
        self.lat += math.sin(t * 0.05) * 0.00002
        self.lon += math.cos(t * 0.04) * 0.00002

        # Wind changes gradually
        self.wind_speed = max(0, self.wind_speed
                               + random.gauss(0, 0.1))
        self.wind_dir   = (self.wind_dir
                           + random.gauss(0, 0.5)) % 360

        # LZ score fluctuates (terrain changes as drone moves)
        self.lz_score = max(0.1, min(1.0,
                        self.lz_score + random.gauss(0, 0.02)))

        # GPS quality degrades if battery low
        if self.battery < 20:
            gps_quality = "WEAK"
            gps_lock    = False
        else:
            gps_quality = "LOCK"
            gps_lock    = True

        # Mission status
        if self.battery < 15:
            status = "EMERGENCY"
        elif self.battery < 25:
            status = "RTB"        # Return to Base
        else:
            status = "ACTIVE"

        self.frame_id += 1

        return TelemetryFrame(
            altitude      = round(self.altitude, 1),
            speed         = round(self.speed, 1),
            heading       = round(self.heading, 1),
            battery       = round(self.battery, 1),
            wind_speed    = round(self.wind_speed, 1),
            wind_dir      = round(self.wind_dir, 1),
            latitude      = round(self.lat, 6),
            longitude     = round(self.lon, 6),
            gps_lock      = gps_lock,
            gps_quality   = gps_quality,
            mission_time  = int(t),
            mission_status= status,
            lz_safe       = self.lz_score > 0.5,
            lz_score      = round(self.lz_score, 2),
            frame_id      = self.frame_id,
            timestamp     = time.time(),
        )


# ── Image Feed Simulator ───────────────────────────────────
# Loops through your dataset images to simulate a live feed.
# Falls back to generating synthetic frames if no dataset found.

class ImageFeedSimulator:
    import sys
    from pathlib import Path
    sys.insert(0, str(Path(__file__).parent.parent))
    from config import SAR_DATA
    
    def __init__(self, dataset_dir=None, fps=2):
        if dataset_dir is None:
            dataset_dir = SAR_DATA
        self.dataset_dir = Path(dataset_dir)
        self.rgb_images     = []
        self.thermal_images = []
        self.fused_images   = []
        self.index          = 0
        self._load_images()

    def _load_images(self):
        """Load all available images from the dataset."""
        # Try fused images first (best quality for display)
        fused_dir = self.dataset_dir / "dataset" / "train" / "images"
        if fused_dir.exists():
            self.fused_images = sorted(fused_dir.glob("*.jpg"))

        # RGB pairs
        rgb_dir = self.dataset_dir / "raw" / "rgb" / "synthetic_pairs"
        if rgb_dir.exists():
            self.rgb_images = sorted(rgb_dir.glob("*_rgb.jpg"))

        # Thermal
        therm_dir = self.dataset_dir / "synthetic" / "thermal"
        if therm_dir.exists():
            self.thermal_images = sorted(therm_dir.glob("*_thermal.jpg"))

        total = max(len(self.fused_images),
                    len(self.rgb_images), 1)
        print(f"[SimEngine] Loaded: {len(self.rgb_images)} RGB, "
              f"{len(self.thermal_images)} thermal, "
              f"{len(self.fused_images)} fused")

    def _make_synthetic_frame(self, frame_type="rgb"):
        """Generate a synthetic frame if no dataset images exist."""
        H, W = 480, 640
        if frame_type == "thermal":
            gray = np.random.randint(30, 180, (H, W), dtype=np.uint8)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            # Add a few heat blobs
            for _ in range(random.randint(1, 3)):
                cx, cy = random.randint(50, W-50), random.randint(50, H-50)
                cv2.ellipse(gray, (cx, cy),
                            (random.randint(8,18), random.randint(15,28)),
                            0, 0, 360, random.randint(180, 230), -1)
            return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        else:
            img = np.random.randint(30, 140, (H, W, 3), dtype=np.uint8)
            return cv2.GaussianBlur(img, (7, 7), 0)

    def next_frame(self):
        """
        Returns (rgb, thermal, fused) numpy arrays for the next frame.
        Loops back to start when all images are exhausted.
        """
        H, W = 480, 640

        # ── RGB ──────────────────────────────────────────
        if self.rgb_images:
            idx = self.index % len(self.rgb_images)
            rgb = cv2.imread(str(self.rgb_images[idx]))
            rgb = cv2.resize(rgb, (W, H)) if rgb is not None \
                  else self._make_synthetic_frame("rgb")
        else:
            rgb = self._make_synthetic_frame("rgb")

        # ── Thermal ───────────────────────────────────────
        if self.thermal_images:
            idx = self.index % len(self.thermal_images)
            thermal = cv2.imread(str(self.thermal_images[idx]))
            thermal = cv2.resize(thermal, (W, H)) if thermal is not None \
                      else self._make_synthetic_frame("thermal")
        else:
            thermal = self._make_synthetic_frame("thermal")

        # ── Fused ─────────────────────────────────────────
        if self.fused_images:
            idx = self.index % len(self.fused_images)
            fused = cv2.imread(str(self.fused_images[idx]))
            fused = cv2.resize(fused, (W, H)) if fused is not None \
                    else cv2.addWeighted(rgb, 0.6, thermal, 0.4, 0)
        else:
            fused = cv2.addWeighted(rgb, 0.6, thermal, 0.4, 0)

        self.index += 1
        return rgb, thermal, fused


# ── Main Sim Engine ────────────────────────────────────────
# Combines image feed + telemetry into complete frames.
# Runs in a background thread, pushing frames into a queue
# that the HMI reads from.

class SimEngine:
    def __init__(self, dataset_dir=r"c:/Users/User/Documents/internship/IISC/Project 1/sar_data", fps=2):
        self.fps          = fps
        self.interval     = 1.0 / fps
        self.running      = False
        self.latest_frame = None          # HMI reads this
        self.lock         = threading.Lock()

        self.telemetry    = TelemetrySimulator()
        self.feed         = ImageFeedSimulator(dataset_dir)
        self._thread      = None

    def start(self):
        """Start the simulation in a background thread."""
        self.running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,             # dies when main app closes
            name="SimEngine"
        )
        self._thread.start()
        print("[SimEngine] Started")

    def stop(self):
        """Stop the simulation."""
        self.running = False
        print("[SimEngine] Stopped")

    def get_latest_frame(self):
        """Thread-safe read of the latest frame."""
        with self.lock:
            return self.latest_frame

    def _run_loop(self):
        """Main loop — generates one frame per tick."""
        while self.running:
            tick_start = time.time()

            # Get next images
            rgb, thermal, fused = self.feed.next_frame()

            # Get telemetry
            telem = self.telemetry.step()

            # Attach images to telemetry frame
            telem.rgb_frame     = rgb
            telem.thermal_frame = thermal
            telem.fused_frame   = fused

            # Store thread-safely
            with self.lock:
                self.latest_frame = telem

            # Sleep to maintain target FPS
            elapsed = time.time() - tick_start
            sleep_t = max(0, self.interval - elapsed)
            time.sleep(sleep_t)


# ── Standalone test ───────────────────────────────────────
if __name__ == "__main__":
    print("Testing SimEngine for 5 seconds...")
    engine = SimEngine(fps=2)
    engine.start()
    for i in range(10):
        time.sleep(0.5)
        frame = engine.get_latest_frame()
        if frame:
            print(f"  Frame {frame.frame_id:>3} | "
                  f"ALT: {frame.altitude:>6.1f}m | "
                  f"BAT: {frame.battery:>5.1f}% | "
                  f"GPS: {frame.latitude:.4f},{frame.longitude:.4f} | "
                  f"Status: {frame.mission_status}")
    engine.stop()
    print("SimEngine test complete.")
