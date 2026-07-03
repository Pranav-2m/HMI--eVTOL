import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import math
import argparse
import sys
from pathlib import Path
from collections import deque

# ── Import our engines ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAR_DATA, find_best_model, setup_path
setup_path()
from step5a_sim_engine       import SimEngine
from fire_inference_bridge   import CombinedInferenceEngine as InferenceEngine
from step5b_inference_engine import draw_detections
from step5c_decision_engine  import DecisionEngine, ACTION_COLORS, URGENCY_COLORS


# ── Theme ─────────────────────────────────────────────────
T = {
    "bg":          "#0A0E17",   # near-black background
    "panel":       "#111827",   # panel background
    "border":      "#1F2937",   # panel border
    "accent":      "#00B4D8",   # cyan accent
    "text":        "#E2E8F0",   # primary text
    "text_dim":    "#64748B",   # dimmed text
    "green":       "#00C853",
    "orange":      "#FF6D00",
    "red":         "#FF1744",
    "yellow":      "#FFD600",
    "blue":        "#2979FF",
    "header_bg":   "#0D1B2A",
    "telem_bg":    "#060B13",
}

# ── Constants ─────────────────────────────────────────────
FEED_W, FEED_H   = 480, 320    # live feed display size
MAP_W,  MAP_H    = 320, 320    # map panel size
UPDATE_MS        = 500         # HMI refresh rate (ms)
MAX_ALERTS       = 8           # max rows in alert list
LOG_LINES        = 6           # lines in decision log


# ── Utility ───────────────────────────────────────────────

def cv2_to_tk(cv_img, w, h):
    """Converts OpenCV BGR image to Tkinter PhotoImage."""
    resized = cv2.resize(cv_img, (w, h))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)


def fmt_time(seconds):
    """Formats seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ── Main Application ───────────────────────────────────────

class CockpitHMI:
    def __init__(self, root, dataset_dir="sar_data", model_path=None):
        self.root = root
        self.root.title("SAR COCKPIT — eVTOL Mission Interface")
        self.root.configure(bg=T["bg"])
        self.root.resizable(False, False)

        # ── State ─────────────────────────────────────────
        self.feed_mode        = "FUSED"     # RGB / THERMAL / FUSED
        self.operator_action  = None        # last operator decision
        self.alert_log        = deque(maxlen=50)
        self.decision_log     = deque(maxlen=LOG_LINES)
        self.survivor_markers = []          # list of (x,y,priority,label)
        self.last_decision    = None
        self.frame_count      = 0
        self._tk_images       = {}          # prevent GC of PhotoImages

        # ── Engines ───────────────────────────────────────
        self.sim       = SimEngine(dataset_dir=dataset_dir, fps=2)
        self.inference = InferenceEngine(model_path=model_path)
        self.decision  = DecisionEngine()

        # ── Build UI ──────────────────────────────────────
        self._build_ui()

        # ── Start engines ─────────────────────────────────
        self.sim.start()

        # ── Start update loop ─────────────────────────────
        self.root.after(UPDATE_MS, self._update_loop)
        self._log("System initialised. Mission ACTIVE.")

    # ══════════════════════════════════════════════════════
    #  UI BUILDER
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        """Builds the full cockpit layout."""
        # ── Title bar ─────────────────────────────────────
        self._build_titlebar()

        # ── Main content area ─────────────────────────────
        main = tk.Frame(self.root, bg=T["bg"])
        main.pack(fill="both", expand=True, padx=6, pady=(0,4))

        # Top row: Feed | Map | Status
        top = tk.Frame(main, bg=T["bg"])
        top.pack(fill="x")

        self._build_feed_panel(top)
        self._build_map_panel(top)
        self._build_status_panel(top)

        # Bottom row: Alerts | Recommendation
        bot = tk.Frame(main, bg=T["bg"])
        bot.pack(fill="x", pady=(4,0))

        self._build_alert_panel(bot)
        self._build_recommendation_panel(bot)

        # Telemetry bar
        self._build_telem_bar()

    def _panel(self, parent, title, w=None, h=None, side="left"):
        """Creates a standard dark panel with title."""
        outer = tk.Frame(parent, bg=T["border"], padx=1, pady=1)
        outer.pack(side=side, padx=3, pady=2,
                   fill="both", expand=(w is None))

        inner = tk.Frame(outer, bg=T["panel"],
                         width=w, height=h)
        inner.pack(fill="both", expand=True)
        if w: inner.pack_propagate(False)

        # Panel header
        hdr = tk.Frame(inner, bg=T["header_bg"], height=22)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=title,
                 bg=T["header_bg"], fg=T["accent"],
                 font=("Courier", 9, "bold")).pack(side="left", padx=6)

        body = tk.Frame(inner, bg=T["panel"])
        body.pack(fill="both", expand=True, padx=4, pady=4)

        return body

    # ── Title Bar ─────────────────────────────────────────

    def _build_titlebar(self):
        bar = tk.Frame(self.root, bg=T["header_bg"], height=36)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Label(bar, text="🚁  SAR COCKPIT",
                 bg=T["header_bg"], fg=T["accent"],
                 font=("Courier", 13, "bold")).pack(side="left", padx=10)

        # Mission status badge
        self._mission_badge = tk.Label(
            bar, text="● MISSION: ACTIVE",
            bg=T["header_bg"], fg=T["green"],
            font=("Courier", 10, "bold")
        )
        self._mission_badge.pack(side="left", padx=20)

        # Mode indicator (MOCK / LIVE)
        mock = getattr(self.inference, "mock_mode", False)
        mode = "MOCK" if mock else "LIVE MODEL"
        tk.Label(bar, text=f"[{mode}]",
                 bg=T["header_bg"], fg=T["text_dim"],
                 font=("Courier", 9)).pack(side="left", padx=10)

        # Clock
        self._clock_lbl = tk.Label(
            bar, text="00:00",
            bg=T["header_bg"], fg=T["text"],
            font=("Courier", 11)
        )
        self._clock_lbl.pack(side="right", padx=10)

    # ── Feed Panel ────────────────────────────────────────

    def _build_feed_panel(self, parent):
        body = self._panel(parent, "📷  LIVE FEED", w=FEED_W+8, side="left")

        # Feed canvas
        self._feed_canvas = tk.Label(
            body, bg="#000000",
            width=FEED_W, height=FEED_H
        )
        self._feed_canvas.pack()

        # Mode toggle buttons
        btn_row = tk.Frame(body, bg=T["panel"])
        btn_row.pack(fill="x", pady=(4,0))

        for mode in ["RGB", "THERMAL", "FUSED"]:
            b = tk.Button(
                btn_row, text=mode,
                bg=T["accent"] if mode == self.feed_mode else T["border"],
                fg=T["bg"] if mode == self.feed_mode else T["text_dim"],
                font=("Courier", 8, "bold"),
                relief="flat", cursor="hand2",
                padx=8, pady=2,
                command=lambda m=mode: self._set_feed_mode(m)
            )
            b.pack(side="left", padx=2)
            self._feed_btns = getattr(self, "_feed_btns", {})
            self._feed_btns[mode] = b

    def _set_feed_mode(self, mode):
        self.feed_mode = mode
        for m, b in self._feed_btns.items():
            if m == mode:
                b.config(bg=T["accent"], fg=T["bg"])
            else:
                b.config(bg=T["border"], fg=T["text_dim"])

    # ── Map Panel ─────────────────────────────────────────

    def _build_map_panel(self, parent):
        body = self._panel(parent, "📍  SURVIVOR MAP",
                           w=MAP_W+8, side="left")

        self._map_canvas = tk.Canvas(
            body, width=MAP_W, height=MAP_H,
            bg="#0D1F0D", highlightthickness=0
        )
        self._map_canvas.pack()
        self._draw_map_grid()

        # Legend
        leg = tk.Frame(body, bg=T["panel"])
        leg.pack(fill="x", pady=(3,0))
        for label, color in [("Critical","#FF1744"),
                              ("High","#FF6D00"),
                              ("Clear","#00C853")]:
            tk.Label(leg, text=f"● {label}",
                     bg=T["panel"], fg=color,
                     font=("Courier", 8)).pack(side="left", padx=6)

    def _draw_map_grid(self):
        """Draws compass grid on the map canvas."""
        c = self._map_canvas
        # Grid lines
        for i in range(0, MAP_W, 40):
            c.create_line(i, 0, i, MAP_H, fill="#1A2E1A", width=1)
        for i in range(0, MAP_H, 40):
            c.create_line(0, i, MAP_W, i, fill="#1A2E1A", width=1)
        # Compass labels
        for label, x, y in [("N",MAP_W//2,8), ("S",MAP_W//2,MAP_H-8),
                              ("E",MAP_W-8,MAP_H//2), ("W",8,MAP_H//2)]:
            c.create_text(x, y, text=label, fill="#2D4A2D",
                          font=("Courier", 8, "bold"))
        # Centre drone marker
        cx, cy = MAP_W//2, MAP_H//2
        c.create_oval(cx-6, cy-6, cx+6, cy+6,
                      fill=T["accent"], outline="white", width=1,
                      tags="drone")
        c.create_text(cx, cy-14, text="🚁",
                      font=("Courier", 10), tags="drone_label")

    # ── Mission Status Panel ───────────────────────────────

    def _build_status_panel(self, parent):
        body = self._panel(parent, "⚡  MISSION STATUS",
                           w=180, side="left")

        fields = [
            ("ALTITUDE",  "alt_lbl",  "-- m"),
            ("SPEED",     "spd_lbl",  "-- km/h"),
            ("HEADING",   "hdg_lbl",  "--°"),
            ("BATTERY",   "bat_lbl",  "-- %"),
            ("WIND",      "wnd_lbl",  "-- kn"),
            ("LZ SCORE",  "lz_lbl",   "--"),
            ("GPS",       "gps_lbl",  "--"),
            ("MISSION T", "mt_lbl",   "00:00"),
            ("SURVIVORS", "sur_lbl",  "0"),
        ]

        for label, attr, default in fields:
            row = tk.Frame(body, bg=T["panel"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{label}:",
                     bg=T["panel"], fg=T["text_dim"],
                     font=("Courier", 8), width=9,
                     anchor="w").pack(side="left")
            lbl = tk.Label(row, text=default,
                           bg=T["panel"], fg=T["text"],
                           font=("Courier", 9, "bold"),
                           anchor="w")
            lbl.pack(side="left")
            setattr(self, f"_{attr}", lbl)

    # ── Alert Panel ───────────────────────────────────────

    def _build_alert_panel(self, parent):
        body = self._panel(parent, "🚨  ALERTS",
                           w=300, side="left")

        # Alert list (using a canvas for coloured rows)
        self._alert_frame = tk.Frame(body, bg=T["panel"])
        self._alert_frame.pack(fill="both", expand=True)

        # Pre-create alert rows
        self._alert_rows = []
        for _ in range(MAX_ALERTS):
            row = tk.Frame(self._alert_frame,
                           bg=T["panel"], height=22)
            row.pack(fill="x", pady=1)
            row.pack_propagate(False)
            lbl = tk.Label(row, text="",
                           bg=T["panel"], fg=T["text_dim"],
                           font=("Courier", 8), anchor="w")
            lbl.pack(fill="x", padx=4)
            self._alert_rows.append((row, lbl))

    # ── Recommendation Panel ───────────────────────────────

    def _build_recommendation_panel(self, parent):
        body = self._panel(parent, "🤖  AI RECOMMENDATION",
                           side="left")

        # Recommendation text
        self._rec_text = tk.Label(
            body,
            text="Initialising...",
            bg=T["panel"], fg=T["text"],
            font=("Courier", 9),
            wraplength=420, justify="left",
            anchor="nw"
        )
        self._rec_text.pack(fill="x", pady=(2,8))

        # Action badge
        self._action_badge = tk.Label(
            body, text="ACTION: HOLD",
            bg=T["blue"], fg="white",
            font=("Courier", 11, "bold"),
            padx=10, pady=4
        )
        self._action_badge.pack(side="left", padx=(0,8))

        # Override buttons
        btn_cfg = dict(
            font=("Courier", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=4
        )
        tk.Button(body, text="✅ ACCEPT",
                  bg=T["green"], fg="black",
                  command=self._on_accept,
                  **btn_cfg).pack(side="left", padx=4)

        tk.Button(body, text="✋ OVERRIDE",
                  bg=T["orange"], fg="black",
                  command=self._on_override,
                  **btn_cfg).pack(side="left", padx=4)

        tk.Button(body, text="⛔ ABORT",
                  bg=T["red"], fg="white",
                  command=self._on_abort,
                  **btn_cfg).pack(side="left", padx=4)

        # Operator log
        self._op_log = tk.Label(
            body, text="",
            bg=T["panel"], fg=T["text_dim"],
            font=("Courier", 8), anchor="w"
        )
        self._op_log.pack(fill="x", pady=(6,0))

    # ── Telemetry Bar ─────────────────────────────────────

    def _build_telem_bar(self):
        bar = tk.Frame(self.root, bg=T["telem_bg"], height=24)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._telem_bar = tk.Label(
            bar, text="Initialising telemetry...",
            bg=T["telem_bg"], fg=T["text_dim"],
            font=("Courier", 8)
        )
        self._telem_bar.pack(side="left", padx=10)

    # ══════════════════════════════════════════════════════
    #  UPDATE LOOP
    # ══════════════════════════════════════════════════════

    def _update_loop(self):
        """Main update — called every UPDATE_MS milliseconds."""
        try:
            frame = self.sim.get_latest_frame()
            if frame is not None:
                self.frame_count += 1

                # 1. Run inference
                detections = self.inference.run(
                    frame.fused_frame, frame.thermal_frame
                )

                # 2. Run decision engine
                decision = self.decision.evaluate(detections, frame)
                self.last_decision = decision

                # 3. Update all panels
                self._update_feed(frame, detections)
                self._update_map(detections, frame)
                self._update_status(frame, decision)
                self._update_alerts(detections)
                self._update_recommendation(decision)
                self._update_telem_bar(frame)
                self._update_clock(frame)

        except Exception as e:
            self._log(f"Update error: {e}")

        # Schedule next update
        self.root.after(UPDATE_MS, self._update_loop)

    # ── Panel Updaters ────────────────────────────────────

    def _update_feed(self, frame, detections):
        """Updates live feed with detection overlay."""
        if self.feed_mode == "RGB":
            img = frame.rgb_frame
        elif self.feed_mode == "THERMAL":
            img = frame.thermal_frame
        else:
            img = frame.fused_frame

        if img is None:
            return

        # Draw detections
        annotated = draw_detections(img.copy(), detections)

        # Convert and display
        tk_img = cv2_to_tk(annotated, FEED_W, FEED_H)
        self._feed_canvas.configure(image=tk_img)
        self._tk_images["feed"] = tk_img  # prevent GC

    def _update_map(self, detections, frame):
        #Updates survivor map with detection markers.
        c = self._map_canvas
        # Clear old markers (keep grid and drone)
        c.delete("marker")

        # Plot each detection as a map dot
        # Convert frame position to map coords
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Normalise bbox centre to frame size
            norm_x = ((x1 + x2) / 2) / 640
            norm_y = ((y1 + y2) / 2) / 480

            # Map to canvas coords (with padding)
            pad   = 30
            map_x = int(pad + norm_x * (MAP_W - 2*pad))
            map_y = int(pad + norm_y * (MAP_H - 2*pad))

            # Colour by priority
            colors = {1: "#FF1744", 2: "#FF6D00", 3: "#FFD600"}
            col = colors.get(det.priority, "#FFFFFF")

            # Pulsing ring (bigger for critical)
            r = 8 if det.priority == 1 else 6
            c.create_oval(map_x-r, map_y-r,
                          map_x+r, map_y+r,
                          outline=col, width=2,
                          fill="", tags="marker")
            c.create_oval(map_x-3, map_y-3,
                          map_x+3, map_y+3,
                          fill=col, outline="",
                          tags="marker")

            # Label
            short = det.class_name[:3].upper()
            c.create_text(map_x, map_y-14,
                          text=short, fill=col,
                          font=("Courier", 7, "bold"),
                          tags="marker")

        # Move drone to current GPS-derived position
        # (slight drift over time)
        t = time.time() % 20
        dx = int(math.sin(t * 0.3) * 15)
        dy = int(math.cos(t * 0.2) * 10)
        cx, cy = MAP_W//2 + dx, MAP_H//2 + dy
        c.coords("drone", cx-6, cy-6, cx+6, cy+6)
        c.coords("drone_label", cx, cy-14)

    def _update_status(self, frame, decision):
        """Updates mission status panel."""
        bat = frame.battery
        bat_col = (T["green"] if bat > 50 else
                   T["orange"] if bat > 25 else T["red"])

        lz_col = T["green"] if frame.lz_safe else T["red"]

        self._alt_lbl.config(text=f"{frame.altitude:.1f} m")
        self._spd_lbl.config(text=f"{frame.speed:.1f} km/h")
        self._hdg_lbl.config(text=f"{frame.heading:.1f}°")
        self._bat_lbl.config(text=f"{bat:.1f} %", fg=bat_col)
        self._wnd_lbl.config(
            text=f"{frame.wind_speed:.1f} kn / {frame.wind_dir:.0f}°"
        )
        self._lz_lbl.config(
            text=f"{frame.lz_score:.0%}", fg=lz_col
        )
        self._gps_lbl.config(
            text=frame.gps_quality,
            fg=T["green"] if frame.gps_lock else T["red"]
        )
        self._mt_lbl.config(text=fmt_time(frame.mission_time))
        self._sur_lbl.config(
            text=str(decision.survivor_count),
            fg=T["red"] if decision.survivor_count > 0 else T["text"]
        )

        # Mission badge
        status_colors = {
            "ACTIVE":    T["green"],
            "RTB":       T["orange"],
            "HOLD":      T["blue"],
            "EMERGENCY": T["red"],
        }
        col = status_colors.get(frame.mission_status, T["text"])
        self._mission_badge.config(
            text=f"● MISSION: {frame.mission_status}",
            fg=col
        )

    def _update_alerts(self, detections):
        """Updates alert list sorted by priority."""
        priority_colors = {
            "CRITICAL": T["red"],
            "HIGH":     T["orange"],
            "MEDIUM":   T["yellow"],
        }

        # Clear all rows first
        for row, lbl in self._alert_rows:
            row.config(bg=T["panel"])
            lbl.config(text="", bg=T["panel"], fg=T["text_dim"])

        # Fill with current detections
        for i, det in enumerate(detections[:MAX_ALERTS]):
            row, lbl = self._alert_rows[i]
            col = priority_colors.get(det.priority_label, T["text"])
            row_bg = "#1A0A0A" if det.priority == 1 else T["panel"]
            row.config(bg=row_bg)
            text = (f"[{det.priority_label[:4]}] "
                    f"{det.class_name:<16} "
                    f"{det.location_hint:<10} "
                    f"{det.confidence:.0%}")
            lbl.config(text=text, bg=row_bg, fg=col)

    def _update_recommendation(self, decision):
        """Updates AI recommendation panel."""
        self._rec_text.config(text=decision.recommendation)

        action_bg = ACTION_COLORS.get(decision.action, T["blue"])
        urgency_fg = URGENCY_COLORS.get(decision.urgency, T["text"])

        self._action_badge.config(
            text=f"ACTION: {decision.action}",
            bg=action_bg,
            fg="black" if decision.action != "RTB" else "white"
        )

    def _update_telem_bar(self, frame):
        """Updates bottom telemetry bar."""
        text = (
            f"  SPD: {frame.speed:.0f}km/h"
            f"  |  HDG: {frame.heading:.0f}°"
            f"  |  WIND: {frame.wind_speed:.0f}kn/{frame.wind_dir:.0f}°"
            f"  |  GPS: {frame.latitude:.4f}, {frame.longitude:.4f}"
            f"  |  [{frame.gps_quality}]"
            f"  |  FRAME: {frame.frame_id}"
        )
        self._telem_bar.config(text=text)

    def _update_clock(self, frame):
        """Updates mission clock in title bar."""
        self._clock_lbl.config(text=fmt_time(frame.mission_time))

    # ══════════════════════════════════════════════════════
    #  OPERATOR CONTROLS
    # ══════════════════════════════════════════════════════

    def _on_accept(self):
        """Operator accepts the AI recommendation."""
        if self.last_decision:
            msg = f"Operator ACCEPTED: {self.last_decision.action}"
            self._op_log.config(text=f"✅ {msg}", fg=T["green"])
            self._log(msg)

    def _on_override(self):
        """Operator overrides the AI recommendation."""
        if self.last_decision:
            msg = f"Operator OVERRODE: {self.last_decision.action}"
            self._op_log.config(text=f"✋ {msg}", fg=T["orange"])
            self._log(msg)
            # In real system: open dialog for manual action entry

    def _on_abort(self):
        """Operator aborts current mission action."""
        self._op_log.config(
            text="⛔ ABORT issued — returning to HOLD", fg=T["red"]
        )
        self._log("Operator ABORT issued")

    def _log(self, msg):
        """Adds a timestamped entry to the internal log."""
        ts = time.strftime("%H:%M:%S")
        self.alert_log.append(f"[{ts}] {msg}")

    # ── Shutdown ──────────────────────────────────────────

    def on_close(self):
        """Clean shutdown."""
        self.sim.stop()
        self.root.destroy()


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SAR eVTOL Cockpit HMI"
    )
    parser.add_argument("--dataset", type=str, default=str(SAR_DATA),
        help="Path to sar_data directory")
    parser.add_argument("--model",   type=str, default=str(find_best_model("human_detector_real") or find_best_model("rgb_detector") or ""),
        help="Path to trained best.pt model file")
    args = parser.parse_args()

    root = tk.Tk()
    root.configure(bg=T["bg"])

    app = CockpitHMI(
        root,
        dataset_dir = args.dataset,
        model_path  = args.model
    )
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()