import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import pyttsx3
import os
import time

try:
    from pyfirmata import Arduino
    FIRMATA_OK = True
except ImportError:
    FIRMATA_OK = False

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["false_smut", "sheath_blight", "sheath_rot", "stem_borer"]

REMEDIES = {
    "false_smut":    "Apply fungicide (Propiconazole) at booting & heading stage. Use certified disease-free seeds.",
    "sheath_blight": "Apply Validamycin 3L @ 2mL/L. Reduce plant density and avoid over-irrigation.",
    "sheath_rot":    "Use certified seeds. Spray Carbendazim 50WP @ 1g/L at boot leaf stage.",
    "stem_borer":    "Apply Cartap Hydrochloride 4G @ 18kg/ha. Release Trichogramma parasitoids.",
}

DISEASE_COLORS = {
    "false_smut":    "#F59E0B",
    "sheath_blight": "#EF4444",
    "sheath_rot":    "#8B5CF6",
    "stem_borer":    "#10B981",
}

# Camera display size (fixed)
CAM_W, CAM_H = 440, 330

os.makedirs("outputs", exist_ok=True)


# ──────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────
class AgriBot:
    def __init__(self, root):
        self.root = root
        self.root.title("AgriBot — Rice Disease Detector + RC Car Controller")
        # width  = 300 (col1) + 2 (div) + 310 (col2) + 2 (div) + 460 (col3) + paddings ≈ 1130
        self.root.geometry("1130x740")
        self.root.resizable(False, False)
        self.root.configure(bg="#0F172A")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Arduino state
        self.board = None
        self.enA = self.enB = None
        self.in1 = self.in2 = self.in3 = self.in4 = None
        self.active_keys = set()

        # Camera / YOLO state
        self.cap          = None
        self.yolo_model   = None
        self.cam_running  = False
        self._last_dets   = []
        self._yolo_tick   = 0
        self._last_spoken = ""
        self._speak_time  = 0

        # TTS
        try:
            self.tts = pyttsx3.init()
        except Exception:
            self.tts = None

        self._build_ui()
        self._bind_keys()

    # ══════════════════════════════════════════════════════════
    # LAYOUT
    # ══════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────
        top = tk.Frame(self.root, bg="#020617", height=48)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)
        tk.Label(top, text="🌾  AGRIBOT CONTROL STATION",
                 font=("Courier New", 14, "bold"), fg="#34D399",
                 bg="#020617").pack(side="left", padx=18, pady=10)
        tk.Label(top, text="YOLOv12 Rice Disease Detection  ·  4WD Arduino Navigation",
                 font=("Courier New", 8), fg="#475569",
                 bg="#020617").pack(side="right", padx=18)

        # ── Body row ─────────────────────────────────────────
        body = tk.Frame(self.root, bg="#0F172A")
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # ── COL 1: Controls (300 px) ─────────────────────────
        col1 = tk.Frame(body, bg="#0F172A", width=300)
        col1.pack(side="left", fill="y")
        col1.pack_propagate(False)
        self._build_controls(col1)

        # divider
        tk.Frame(body, bg="#1E293B", width=2).pack(side="left", fill="y", padx=8)

        # ── COL 2: Remedy (310 px) ───────────────────────────
        col2 = tk.Frame(body, bg="#0F172A", width=310)
        col2.pack(side="left", fill="y")
        col2.pack_propagate(False)
        self._build_remedy_col(col2)

        # divider
        tk.Frame(body, bg="#1E293B", width=2).pack(side="left", fill="y", padx=8)

        # ── COL 3: Camera (460 px) ───────────────────────────
        col3 = tk.Frame(body, bg="#0F172A", width=460)
        col3.pack(side="left", fill="y")
        col3.pack_propagate(False)
        self._build_camera_col(col3)

    # ══════════════════════════════════════════════════════════
    # COL 1 — CONTROLS
    # ══════════════════════════════════════════════════════════
    def _build_controls(self, parent):
        self._section(parent, "⚡  ARDUINO CONNECTION", self._ui_connection).pack(fill="x", pady=(0, 8))
        self._section(parent, "🎛  MOTOR SPEED",        self._ui_speed).pack(fill="x", pady=(0, 8))
        self._section(parent, "🕹  NAVIGATION",         self._ui_nav).pack(fill="x", pady=(0, 8))
        self._section(parent, "⌨  KEYBOARD SHORTCUTS",  self._ui_keys).pack(fill="x")

    def _ui_connection(self, f):
        row = tk.Frame(f, bg="#1E293B")
        row.pack(fill="x")
        tk.Label(row, text="COM Port:", font=("Courier New", 9),
                 fg="#94A3B8", bg="#1E293B").pack(side="left")
        self.port_var = tk.StringVar(value="COM3")
        tk.Entry(row, textvariable=self.port_var, width=7,
                 font=("Courier New", 10, "bold"), bg="#0F172A",
                 fg="#E2E8F0", insertbackground="white",
                 relief="flat").pack(side="left", padx=6, ipady=4)
        self.conn_btn = tk.Button(
            row, text="CONNECT", font=("Courier New", 8, "bold"),
            bg="#34D399", fg="#0F172A", relief="flat",
            activebackground="#6EE7B7", cursor="hand2",
            command=self.connect_arduino, padx=8, pady=3)
        self.conn_btn.pack(side="left", padx=(6, 0))
        self.status_lbl = tk.Label(
            f, text="● NOT CONNECTED",
            font=("Courier New", 8, "bold"), fg="#EF4444", bg="#1E293B")
        self.status_lbl.pack(anchor="w", pady=(6, 0))

    def _ui_speed(self, f):
        self.speed_var = tk.IntVar(value=80)
        self.speed_disp = tk.Label(f, text="80%",
                                   font=("Courier New", 24, "bold"),
                                   fg="#34D399", bg="#1E293B")
        self.speed_disp.pack()
        tk.Scale(f, from_=0, to=100, orient="horizontal", length=260,
                 variable=self.speed_var, command=self._on_speed,
                 bg="#1E293B", fg="#94A3B8", troughcolor="#334155",
                 highlightthickness=0, relief="flat",
                 activebackground="#34D399").pack(padx=4)

    def _ui_nav(self, f):
        g = tk.Frame(f, bg="#1E293B")
        g.pack(pady=4)
        self._nav_btn(g, "▲\nFWD",  "#38BDF8", 0, 1)
        self._nav_btn(g, "◀\nLEFT", "#A78BFA", 1, 0)
        self._nav_btn(g, "■\nSTOP", "#EF4444", 1, 1)
        self._nav_btn(g, "▶\nRGT",  "#A78BFA", 1, 2)
        self._nav_btn(g, "▼\nBWD",  "#38BDF8", 2, 1)

    def _nav_btn(self, parent, text, color, row, col):
        b = tk.Button(parent, text=text, font=("Courier New", 9, "bold"),
                      width=7, height=3, bg="#0F172A", fg=color,
                      activebackground=color, activeforeground="#0F172A",
                      relief="flat", cursor="hand2", bd=0)
        b.grid(row=row, column=col, padx=4, pady=4)
        actions = {
            "▲\nFWD":  (self.move_fwd,   self.stop_motors),
            "▼\nBWD":  (self.move_bwd,   self.stop_motors),
            "◀\nLEFT": (self.move_left,  self.stop_motors),
            "▶\nRGT":  (self.move_rgt,   self.stop_motors),
            "■\nSTOP": (self.stop_motors, None),
        }
        press, release = actions[text]
        b.bind("<ButtonPress-1>",   lambda e: press())
        if release:
            b.bind("<ButtonRelease-1>", lambda e: release())

    def _ui_keys(self, f):
        for k, v in [("W / ↑", "Forward"), ("S / ↓", "Backward"),
                     ("A / ←", "Left"),    ("D / →", "Right")]:
            r = tk.Frame(f, bg="#0F172A")
            r.pack(fill="x", pady=1)
            tk.Label(r, text=k, font=("Courier New", 9, "bold"),
                     fg="#38BDF8", bg="#0F172A", width=10, anchor="w").pack(side="left", padx=6)
            tk.Label(r, text=v, font=("Courier New", 8),
                     fg="#64748B", bg="#0F172A").pack(side="left")

    # ══════════════════════════════════════════════════════════
    # COL 2 — REMEDY ADVISOR
    # ══════════════════════════════════════════════════════════
    def _build_remedy_col(self, parent):
        outer = tk.Frame(parent, bg="#1E293B")
        outer.pack(fill="both", expand=True)

        hdr = tk.Frame(outer, bg="#020617", height=28)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="💊  DISEASE REMEDY ADVISOR",
                 font=("Courier New", 8), fg="#64748B",
                 bg="#020617").pack(side="left", padx=10, pady=4)

        # Detection status badge
        self.det_badge = tk.Label(
            outer, text="NO DISEASE DETECTED",
            font=("Courier New", 9, "bold"), fg="#475569",
            bg="#1E293B", pady=6)
        self.det_badge.pack(fill="x", padx=8)

        # Separator line
        tk.Frame(outer, bg="#334155", height=1).pack(fill="x", padx=8, pady=(0, 6))

        # Scrollable-style remedy content frame
        self.remedy_frame = tk.Frame(outer, bg="#1E293B")
        self.remedy_frame.pack(fill="both", expand=True, padx=10, pady=4)
        self._remedy_clear()

    # ══════════════════════════════════════════════════════════
    # COL 3 — CAMERA
    # ══════════════════════════════════════════════════════════
    def _build_camera_col(self, parent):
        outer = tk.Frame(parent, bg="#1E293B")
        outer.pack(fill="both", expand=True)

        # Header
        hdr = tk.Frame(outer, bg="#020617", height=28)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="📷  LIVE FEED — Rice Disease Scanner",
                 font=("Courier New", 8), fg="#64748B",
                 bg="#020617").pack(side="left", padx=10, pady=4)

        # Controls row
        ctrl = tk.Frame(outer, bg="#1E293B")
        ctrl.pack(fill="x", padx=8, pady=6)

        tk.Label(ctrl, text="Cam:", font=("Courier New", 8),
                 fg="#94A3B8", bg="#1E293B").pack(side="left")
        self.cam_idx = tk.Entry(ctrl, width=3, font=("Courier New", 9),
                                bg="#0F172A", fg="#E2E8F0",
                                insertbackground="white", relief="flat")
        self.cam_idx.insert(0, "0")
        self.cam_idx.pack(side="left", padx=(4, 8), ipady=3)

        self.cam_btn = tk.Button(
            ctrl, text="▶ START CAM", font=("Courier New", 8, "bold"),
            bg="#34D399", fg="#0F172A", relief="flat",
            activebackground="#6EE7B7", cursor="hand2",
            command=self.toggle_cam, padx=8, pady=3)
        self.cam_btn.pack(side="left")

        self.upload_btn = tk.Button(
            ctrl, text="🖼 UPLOAD", font=("Courier New", 8, "bold"),
            bg="#38BDF8", fg="#0F172A", relief="flat",
            activebackground="#7DD3FC", cursor="hand2",
            command=self.upload_image, padx=8, pady=3)
        self.upload_btn.pack(side="left", padx=(6, 0))

        # Fixed-size canvas  440 × 330
        cam_bg = tk.Frame(outer, bg="#020617",
                          width=CAM_W, height=CAM_H)
        cam_bg.pack(padx=8, pady=(0, 6))
        cam_bg.pack_propagate(False)

        self.canvas = tk.Canvas(cam_bg, width=CAM_W, height=CAM_H,
                                bg="#020617", highlightthickness=0)
        self.canvas.pack()
        self._canvas_placeholder("[ Click ▶ START CAM to begin ]")

        # FPS / status bar
        self.fps_lbl = tk.Label(outer, text="",
                                font=("Courier New", 7), fg="#334155",
                                bg="#1E293B")
        self.fps_lbl.pack(anchor="w", padx=10)

    def _canvas_placeholder(self, msg):
        self.canvas.delete("all")
        self.canvas.create_text(CAM_W // 2, CAM_H // 2,
                                text=msg, fill="#334155",
                                font=("Courier New", 9), anchor="center")

    # ══════════════════════════════════════════════════════════
    # SECTION HELPER
    # ══════════════════════════════════════════════════════════
    def _section(self, parent, title, builder):
        outer = tk.Frame(parent, bg="#1E293B")
        hdr   = tk.Frame(outer, bg="#020617", height=26)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=title, font=("Courier New", 8),
                 fg="#64748B", bg="#020617").pack(side="left", padx=10, pady=3)
        inner = tk.Frame(outer, bg="#1E293B", padx=8, pady=8)
        inner.pack(fill="both", expand=True)
        builder(inner)
        return outer

    # ══════════════════════════════════════════════════════════
    # ARDUINO
    # ══════════════════════════════════════════════════════════
    def connect_arduino(self):
        if self.board:
            self._arduino_off(); return
        if not FIRMATA_OK:
            messagebox.showerror("Missing", "pyfirmata not installed.\npip install pyfirmata")
            return
        port = self.port_var.get().strip().upper()
        self.status_lbl.config(text="● CONNECTING…", fg="#F59E0B")
        self.root.update()
        try:
            self.board = Arduino(port)
            self.enB  = self.board.get_pin('d:9:p')
            self.enA  = self.board.get_pin('d:10:p')
            self.in1  = self.board.get_pin('d:6:o')
            self.in2  = self.board.get_pin('d:5:o')
            self.in3  = self.board.get_pin('d:4:o')
            self.in4  = self.board.get_pin('d:3:o')
            self._on_speed(self.speed_var.get())
            self.status_lbl.config(text=f"● CONNECTED  ({port})", fg="#34D399")
            self.conn_btn.config(text="DISCONNECT")
        except Exception as e:
            self.board = None
            self.status_lbl.config(text="● NOT CONNECTED", fg="#EF4444")
            messagebox.showerror("Connection Error", str(e))

    def _arduino_off(self):
        self.stop_motors()
        try:
            self.enA.write(0); self.enB.write(0)
            self.board.exit()
        except Exception:
            pass
        self.board = None
        self.status_lbl.config(text="● NOT CONNECTED", fg="#EF4444")
        self.conn_btn.config(text="CONNECT")

    def _on_speed(self, val):
        self.speed_disp.config(text=f"{int(float(val))}%")
        if self.board:
            v = float(val) / 100.0
            self.enA.write(v); self.enB.write(v)

    def move_left(self):
        if self.board:
            self.in1.write(1); self.in2.write(0)
            self.in3.write(1); self.in4.write(0)

    def move_rgt(self):
        if self.board:
            self.in1.write(0); self.in2.write(1)
            self.in3.write(0); self.in4.write(1)

    def move_fwd(self):
        if self.board:
            self.in1.write(0); self.in2.write(1)
            self.in3.write(1); self.in4.write(0)

    def move_bwd(self):
        if self.board:
            self.in1.write(1); self.in2.write(0)
            self.in3.write(0); self.in4.write(1)

    def stop_motors(self):
        if self.board:
            self.in1.write(0); self.in2.write(0)
            self.in3.write(0); self.in4.write(0)

    # ══════════════════════════════════════════════════════════
    # KEYBOARD
    # ══════════════════════════════════════════════════════════
    def _bind_keys(self):
        self.root.bind("<KeyPress>",   self._kp)
        self.root.bind("<KeyRelease>", self._kr)

    def _kp(self, e):
        k = e.keysym.lower()
        if k in self.active_keys: return
        self.active_keys.add(k)
        {"w": self.move_fwd,  "up":    self.move_fwd,
         "s": self.move_bwd,  "down":  self.move_bwd,
         "a": self.move_left, "left":  self.move_left,
         "d": self.move_rgt,  "right": self.move_rgt}.get(k, lambda: None)()

    def _kr(self, e):
        k = e.keysym.lower()
        self.active_keys.discard(k)
        if not self.active_keys & {"w","up","s","down","a","left","d","right"}:
            self.stop_motors()

    # ══════════════════════════════════════════════════════════
    # CAMERA
    # ══════════════════════════════════════════════════════════
    def toggle_cam(self):
        if self.cam_running:
            # STOP
            self.cam_running = False
            self.cam_btn.config(text="▶ START CAM", bg="#34D399")
            if self.cap:
                self.cap.release(); self.cap = None
            self._canvas_placeholder("[ Camera stopped ]")
            self.fps_lbl.config(text="")
            self.det_badge.config(text="CAM OFF", fg="#475569")
            self._remedy_clear()
        else:
            # START — open camera first
            idx = int(self.cam_idx.get().strip() or "0")
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error",
                    f"Cannot open camera index {idx}.\nTry 0, 1, or 2.")
                self.cap = None; return

            # Load model (non-blocking warning if missing)
            if self.yolo_model is None:
                try:
                    self.yolo_model = YOLO("best.pt")
                except Exception as ex:
                    messagebox.showwarning("Model Warning",
                        f"best.pt not found — camera runs without detection.\n{ex}")

            self.cam_running = True
            self._last_dets  = []
            self._yolo_tick  = 0
            self._t0 = time.time()
            self._frame_count = 0
            self.cam_btn.config(text="■ STOP CAM", bg="#EF4444")
            self._poll()

    def _poll(self):
        if not self.cam_running: return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(40, self._poll); return

        frame = cv2.flip(frame, 1)

        # YOLO every 5 frames
        self._yolo_tick += 1
        if self.yolo_model and self._yolo_tick % 5 == 0:
            try:
                results = self.yolo_model(frame, verbose=False, conf=0.45)
                dets = []
                for r in results:
                    if r.boxes is None: continue
                    for box in r.boxes:
                        cls  = int(box.cls[0])
                        conf = float(box.conf[0])
                        lbl  = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else r.names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        dets.append((lbl, conf, x1, y1, x2, y2))
                self._last_dets = dets
            except Exception:
                pass

        # Draw boxes
        for (lbl, conf, x1, y1, x2, y2) in self._last_dets:
            hx  = DISEASE_COLORS.get(lbl, "#FFFFFF")
            bgr = (int(hx[5:7], 16), int(hx[3:5], 16), int(hx[1:3], 16))
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            txt = f"{lbl} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), bgr, -1)
            cv2.putText(frame, txt, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # FPS counter
        self._frame_count += 1
        elapsed = time.time() - self._t0
        if elapsed > 0:
            fps = self._frame_count / elapsed
            self.fps_lbl.config(text=f"FPS: {fps:.1f}")

        # Render to fixed canvas
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((CAM_W, CAM_H), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # Update remedy col
        if self._last_dets:
            top = max(self._last_dets, key=lambda x: x[1])
            self._update_disease(top[0], top[1])
        else:
            self.det_badge.config(text="✓ HEALTHY", fg="#34D399")
            self._remedy_clear()

        self.root.after(33, self._poll)

    # ══════════════════════════════════════════════════════════
    # IMAGE UPLOAD
    # ══════════════════════════════════════════════════════════
    def upload_image(self):
        if self.cam_running:
            messagebox.showinfo("Info", "Stop the camera first.")
            return
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return

        if self.yolo_model is None:
            try:
                self.yolo_model = YOLO("best.pt")
            except Exception as ex:
                messagebox.showerror("Model Error", f"Cannot load best.pt\n{ex}"); return

        img_cv = cv2.imread(path)
        if img_cv is None:
            messagebox.showerror("Error", "Cannot read image."); return

        results = self.yolo_model(img_cv, verbose=False, conf=0.45)
        dets = []
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                lbl  = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else r.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets.append((lbl, conf, x1, y1, x2, y2))
                hx  = DISEASE_COLORS.get(lbl, "#00FF00")
                bgr = (int(hx[5:7],16), int(hx[3:5],16), int(hx[1:3],16))
                cv2.rectangle(img_cv, (x1,y1),(x2,y2), bgr, 2)
                txt = f"{lbl} {conf:.0%}"
                (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(img_cv,(x1,y1-th-8),(x1+tw+6,y1), bgr,-1)
                cv2.putText(img_cv, txt,(x1+3,y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2)

        cv2.imwrite("outputs/result.jpg", img_cv)

        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((CAM_W, CAM_H), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        if dets:
            top = max(dets, key=lambda x: x[1])
            self._update_disease(top[0], top[1])
        else:
            self.det_badge.config(text="✓ HEALTHY", fg="#34D399")
            self._remedy_clear()

    # ══════════════════════════════════════════════════════════
    # REMEDY PANEL
    # ══════════════════════════════════════════════════════════
    def _update_disease(self, label, conf):
        color = DISEASE_COLORS.get(label, "#F59E0B")
        self.det_badge.config(
            text=f"⚠  {label.replace('_',' ').upper()}  {conf:.0%}",
            fg=color)
        self._remedy_show(label, color)

        # TTS throttled to once per 5s per disease
        now = time.time()
        if self.tts and (label != self._last_spoken or now - self._speak_time > 5):
            self._last_spoken = label
            self._speak_time  = now
            try:
                self.tts.say(label.replace("_", " "))
                self.tts.runAndWait()
            except Exception:
                pass

    def _remedy_show(self, label, color):
        for w in self.remedy_frame.winfo_children():
            w.destroy()

        display = label.replace("_", " ").title()
        remedy  = REMEDIES.get(label, "No remedy found.")

        # Disease name
        tk.Label(self.remedy_frame, text=display,
                 font=("Courier New", 13, "bold"), fg=color,
                 bg="#1E293B", anchor="w").pack(fill="x", pady=(4, 6))

        # Separator
        tk.Frame(self.remedy_frame, bg=color, height=2).pack(fill="x", pady=(0, 8))

        # Label
        tk.Label(self.remedy_frame, text="RECOMMENDED TREATMENT:",
                 font=("Courier New", 7, "bold"), fg="#64748B",
                 bg="#1E293B", anchor="w").pack(fill="x")

        # Remedy text — split by sentence for cleaner display
        for i, sentence in enumerate(remedy.split(". ")):
            sentence = sentence.strip()
            if not sentence: continue
            if not sentence.endswith("."): sentence += "."
            row = tk.Frame(self.remedy_frame, bg="#0F172A")
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{i+1:02d}",
                     font=("Courier New", 9, "bold"), fg=color,
                     bg="#0F172A", width=3).pack(side="left", padx=(4, 6))
            tk.Label(row, text=sentence,
                     font=("Courier New", 8), fg="#94A3B8",
                     bg="#0F172A", wraplength=240,
                     justify="left", anchor="w").pack(side="left", fill="x", pady=4)

    def _remedy_clear(self):
        for w in self.remedy_frame.winfo_children():
            w.destroy()
        tk.Label(self.remedy_frame,
                 text="✅  Healthy",
                 font=("Courier New", 12, "bold"), fg="#34D399",
                 bg="#1E293B").pack(pady=(20, 6))
        tk.Label(self.remedy_frame,
                 text="No disease detected.\nAim camera at rice plants\nor upload an image to scan.",
                 font=("Courier New", 8), fg="#475569",
                 bg="#1E293B", justify="center").pack()

    # ══════════════════════════════════════════════════════════
    # CLEANUP
    # ══════════════════════════════════════════════════════════
    def on_close(self):
        self.cam_running = False
        if self.cap:
            self.cap.release()
        if self.board:
            self.stop_motors()
            try:
                self.enA.write(0); self.enB.write(0)
                self.board.exit()
            except Exception:
                pass
        self.root.destroy()


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = AgriBot(root)
    root.mainloop()
