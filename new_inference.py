# modern_tb_detection_gui.py

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Fix NumPy compatibility issue
import numpy as np
if hasattr(np, '__version__') and tuple(map(int, np.__version__.split('.')[:2])) >= (2, 0):
    print("Warning: NumPy 2.x detected. Consider downgrading for better compatibility.")

import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import time
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid threading issues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageDraw
import io

# Try importing librosa with fallback
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except Exception as e:
    print(f"Warning: librosa import failed: {e}")
    LIBROSA_AVAILABLE = False

# Try scipy import with fallback
try:
    from scipy.ndimage import zoom
    SCIPY_ZOOM_AVAILABLE = True
except Exception as e:
    print(f"Warning: scipy.ndimage.zoom import failed: {e}")  
    SCIPY_ZOOM_AVAILABLE = False

import timm 

# ─── Constants ───────────────────────────────────────────────────────────────
SR = 16000
CROP = 0.5
N_MELS = 224
FMAX = 8000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alternative implementations for compatibility
def simple_zoom(array, zoom_factors, order=1):
    """Simple zoom implementation using numpy only"""
    if not SCIPY_ZOOM_AVAILABLE:
        from numpy import interp
        old_shape = array.shape
        new_shape = tuple(int(old_shape[i] * zoom_factors[i]) for i in range(len(old_shape)))
        
        if len(array.shape) == 2:
            # 2D case - simple nearest neighbor interpolation
            result = np.zeros(new_shape)
            for i in range(new_shape[0]):
                for j in range(new_shape[1]):
                    old_i = int(i * old_shape[0] / new_shape[0])
                    old_j = int(j * old_shape[1] / new_shape[1])
                    old_i = min(old_i, old_shape[0] - 1)
                    old_j = min(old_j, old_shape[1] - 1)
                    result[i, j] = array[old_i, old_j]
            return result
        else:
            return array
    else:
        return zoom(array, zoom_factors, order=order)

def load_audio_fallback(path, sr=16000):
    """Fallback audio loading without librosa"""
    try:
        import soundfile as sf
        y, orig_sr = sf.read(path)
        if orig_sr != sr:
            ratio = sr / orig_sr
            new_length = int(len(y) * ratio)
            y = np.interp(np.linspace(0, len(y), new_length), np.arange(len(y)), y)
        return y
    except ImportError:
        try:
            import wave
            with wave.open(path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                return y
        except Exception as e:
            return None

# ─── Model Components ────────────────────────────────────────────────────────
class TemporalShift(nn.Module):
    def __init__(self, channels, shift_div=8):
        super().__init__()
        self.fold = channels // shift_div

    def forward(self, x):
        B, C, T, F = x.size()
        t = x.permute(0, 2, 1, 3).contiguous()
        out = torch.zeros_like(t)
        out[:, :-1, :self.fold, :] = t[:, 1:, :self.fold, :]
        out[:, 1:, self.fold:2*self.fold, :] = t[:, :-1, self.fold:2*self.fold, :]
        out[:, :, 2*self.fold:, :] = t[:, :, 2*self.fold:, :]
        return out.permute(0, 2, 1, 3)

class Res2TSMBlock(nn.Module):
    def __init__(self, channels, scale=4, shift_div=8):
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        self.scale = scale
        self.width = channels // scale
        self.temporal_shift = TemporalShift(channels, shift_div)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width,
                      kernel_size=(3, 1), padding=(1, 0),
                      groups=self.width, bias=False)
            for _ in range(scale-1)
        ])
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.temporal_shift(x)
        splits = torch.split(x, self.width, dim=1)
        y = splits[0]
        outs = [y]
        for i in range(1, self.scale):
            sp = splits[i] + y
            sp = self.convs[i-1](sp)
            y = sp
            outs.append(sp)
        out = torch.cat(outs, dim=1)
        out = self.bn(out)
        return self.act(out)

class MobileNetV4_Res2TSM(nn.Module):
    def __init__(self, model_key, scale=4, shift_div=8, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_key, pretrained=False, features_only=True)
        C = self.backbone.feature_info.channels()[-1]
        self.res2tsm = Res2TSMBlock(C, scale=scale, shift_div=shift_div)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(C, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = self.res2tsm(feat)
        out = self.global_pool(feat).view(feat.size(0), -1)
        return self.fc(out).squeeze(1)

# ─── Audio Processing Functions ──────────────────────────────────────────────
def load_and_segment(path):
    try:
        if LIBROSA_AVAILABLE:
            try:
                y, _ = librosa.load(path, sr=SR)
            except Exception:
                y = load_audio_fallback(path, sr=SR)
                if y is None:
                    return None
        else:
            y = load_audio_fallback(path, sr=SR)
            if y is None:
                return None
        
        if len(y) == 0:
            return None
        
        # Simple silence trimming
        if LIBROSA_AVAILABLE:
            try:
                y, _ = librosa.effects.trim(y, top_db=20)
            except:
                energy = y**2
                threshold = np.max(energy) * 0.01
                valid_indices = np.where(energy > threshold)[0]
                if len(valid_indices) > 0:
                    y = y[valid_indices[0]:valid_indices[-1]+1]
        else:
            energy = y**2
            threshold = np.max(energy) * 0.01
            valid_indices = np.where(energy > threshold)[0]
            if len(valid_indices) > 0:
                y = y[valid_indices[0]:valid_indices[-1]+1]
        
        target_len = int(SR * CROP)
        if len(y) >= target_len:
            energy = np.convolve(y**2, np.ones(target_len), mode='valid')
            start = np.argmax(energy)
            seg = y[start:start+target_len]
        else:
            seg = np.zeros(target_len, dtype=y.dtype)
            seg[:len(y)] = y
            
        return seg
    except Exception:
        return None

def make_mel_rgb(y_seg):
    try:
        if LIBROSA_AVAILABLE:
            try:
                mel = librosa.feature.melspectrogram(
                    y=y_seg, sr=SR, n_mels=N_MELS, fmax=FMAX,
                    hop_length=512, win_length=2048
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
            except Exception:
                # Simple fallback spectrogram
                n_fft = 2048
                hop_length = 512
                stft = np.abs(np.fft.fft(y_seg.reshape(-1, n_fft), axis=1))
                mel_db = 10 * np.log10(stft[:N_MELS, :224] + 1e-10)
        else:
            # Very basic spectrogram
            n_fft = 2048
            stft = np.abs(np.fft.fft(y_seg.reshape(-1, n_fft), axis=1))
            mel_db = 10 * np.log10(stft[:N_MELS, :224] + 1e-10)
        
        target_shape = (224, 224)
        if mel_db.shape != target_shape:
            zoom_factors = (target_shape[0]/mel_db.shape[0], target_shape[1]/mel_db.shape[1])
            resized = simple_zoom(mel_db, zoom_factors, order=1)
        else:
            resized = mel_db
        
        if np.ptp(resized) > 0:
            normed = (resized - resized.min()) / np.ptp(resized)
        else:
            normed = np.zeros_like(resized)
            
        rgb = np.stack([normed] * 3, axis=0)
        return resized, rgb
    except Exception:
        return None, None

# ─── Modern UI Components ────────────────────────────────────────────────────
class ModernButton(tk.Frame):
    def __init__(self, parent, text, command=None, style="primary", width=None, height=40, state="normal", **kwargs):
        super().__init__(parent, **kwargs)
        
        self.command = command
        self.style = style
        self.state_var = tk.StringVar(value=state)
        
        # Color schemes for different button styles
        self.styles = {
            "primary": {"bg": "#6366f1", "hover": "#4f46e5", "text": "white", "disabled": "#9ca3af"},
            "success": {"bg": "#10b981", "hover": "#059669", "text": "white", "disabled": "#9ca3af"},
            "danger": {"bg": "#ef4444", "hover": "#dc2626", "text": "white", "disabled": "#9ca3af"},
            "secondary": {"bg": "#f3f4f6", "hover": "#e5e7eb", "text": "#374151", "disabled": "#d1d5db"},
            "gradient": {"bg": "#6366f1", "hover": "#4f46e5", "text": "white", "disabled": "#9ca3af"}
        }
        
        # Create canvas for custom drawing
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0, cursor="hand2")
        if width:
            self.canvas.config(width=width)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        
        self.text = text
        self.width = width or 120
        self.height = height
        
        # Bind events
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<Configure>", self._on_configure)
        
        # Animation variables
        self.hover_state = False
        self.draw_button()
        
        # Trace state changes
        self.state_var.trace("w", self._on_state_change)
        
    def set_text_and_style(self, text=None, style=None):
        if text is not None:
            self.text = text
        if style is not None:
            self.style = style
        self.draw_button()
    
    def _on_configure(self, event):
        self.width = event.width
        self.draw_button()
    
    def _on_click(self, event):
        if self.state_var.get() == "normal" and self.command:
            # Add click animation
            self.canvas.create_oval(event.x-10, event.y-10, event.x+10, event.y+10, 
                                  fill="white", outline="", stipple="gray25")
            self.after(100, self.draw_button)
            self.command()
    
    def _on_enter(self, event):
        if self.state_var.get() == "normal":
            self.hover_state = True
            self.draw_button()
    
    def _on_leave(self, event):
        self.hover_state = False
        self.draw_button()
    
    def _on_state_change(self, *args):
        self.draw_button()
        cursor = "hand2" if self.state_var.get() == "normal" else "arrow"
        self.canvas.config(cursor=cursor)
    
    def draw_button(self):
        self.canvas.delete("all")
        
        style_config = self.styles[self.style]
        state = self.state_var.get()
        
        if state == "disabled":
            bg_color = style_config["disabled"]
            text_color = "white"
            border_color = style_config["disabled"]
        elif self.hover_state and state == "normal":
            bg_color = style_config["hover"]
            text_color = style_config["text"]
            border_color = style_config["hover"]
        else:
            bg_color = style_config["bg"]
            text_color = style_config["text"]
            border_color = style_config["bg"]
        
        # Draw rounded rectangle
        radius = 8
        x1, y1, x2, y2 = 0, 0, self.width, self.height
        
        # Create rounded rectangle using multiple shapes
        self.canvas.create_oval(x1, y1, x1+2*radius, y1+2*radius, fill=bg_color, outline=border_color, width=1)
        self.canvas.create_oval(x2-2*radius, y1, x2, y1+2*radius, fill=bg_color, outline=border_color, width=1)
        self.canvas.create_oval(x1, y2-2*radius, x1+2*radius, y2, fill=bg_color, outline=border_color, width=1)
        self.canvas.create_oval(x2-2*radius, y2-2*radius, x2, y2, fill=bg_color, outline=border_color, width=1)
        
        self.canvas.create_rectangle(x1+radius, y1, x2-radius, y2, fill=bg_color, outline="")
        self.canvas.create_rectangle(x1, y1+radius, x2, y2-radius, fill=bg_color, outline="")
        
        # Add gradient effect for primary style
        if self.style == "gradient" and state == "normal":
            for i in range(5):
                alpha = 1 - (i * 0.15)
                color = f"#{hex(int(99*alpha))[2:].zfill(2)}{hex(int(102*alpha))[2:].zfill(2)}{hex(int(241*alpha))[2:].zfill(2)}"
                self.canvas.create_rectangle(x1, y1+i*2, x2, y1+(i+1)*2, fill=color, outline="")
        
        # Draw text
        text_x = self.width // 2
        text_y = self.height // 2
        font_size = 10 if len(self.text) > 15 else 11
        self.canvas.create_text(text_x, text_y, text=self.text, fill=text_color, 
                              font=("Segoe UI", font_size, "bold"), anchor="center")
    
    def configure_state(self, state):
        self.state_var.set(state)

class ModernCard(tk.Frame):
    def __init__(self, parent, title=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Configure card styling
        self.configure(bg="white", relief="flat", bd=0)
        
        # Add shadow effect using multiple frames
        shadow_frame = tk.Frame(parent, bg="#e2e8f0", height=2)
        shadow_frame.place(in_=self, x=2, y=2, relwidth=1, relheight=1)
        self.lift()
        
        if title:
            title_label = tk.Label(self, text=title, font=("Segoe UI", 14, "bold"), 
                                 bg="white", fg="#1f2937")
            title_label.pack(anchor="w", padx=20, pady=(20, 10))

class AnimatedProgressBar(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.configure(bg="white")
        
        self.canvas = tk.Canvas(self, height=8, bg="white", highlightthickness=0)
        self.canvas.pack(fill="x", padx=2, pady=2)
        
        self.progress = 0
        self.animation_id = None
        
    def set_progress(self, value):
        target = max(0, min(100, value))
        self.animate_to(target)
    
    def animate_to(self, target):
        if self.animation_id:
            self.after_cancel(self.animation_id)
        
        def animate():
            diff = target - self.progress
            if abs(diff) < 1:
                self.progress = target
            else:
                self.progress += diff * 0.2
            
            self.draw_progress()
            
            if abs(target - self.progress) > 0.5:
                self.animation_id = self.after(16, animate)  # ~60fps
        
        animate()
    
    def draw_progress(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        if width <= 1:
            width = 300
        
        height = 8
        
        # Background
        self.canvas.create_rectangle(0, 0, width, height, fill="#f3f4f6", outline="")
        
        # Progress
        progress_width = width * (self.progress / 100)
        if progress_width > 0:
            # Gradient effect
            for i in range(int(progress_width)):
                ratio = i / max(1, progress_width)
                color = self.interpolate_color("#6366f1", "#8b5cf6", ratio)
                self.canvas.create_rectangle(i, 0, i+1, height, fill=color, outline="")

    def interpolate_color(self, color1, color2, ratio):
        # Simple color interpolation
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        
        return f"#{r:02x}{g:02x}{b:02x}"

# ─── Main Application ────────────────────────────────────────────────────────
class ModernTBDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CoughDetect AI - TB Detection System")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)
        
        # Modern color palette
        self.colors = {
            'bg': '#f8fafc',
            'card_bg': '#ffffff',
            'primary': '#6366f1',
            'primary_light': '#a5b4fc',
            'success': '#10b981',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'info': '#3b82f6',
            'text': '#1f2937',
            'text_light': '#6b7280',
            'border': '#e5e7eb',
            'shadow': '#f3f4f6'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # App state
        self.model = None
        self.audio_files = []
        self.results = []
        self.recorded_coughs = []
        self.required_coughs = 5
        self.is_recording = False
        self.recording_thread = None
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        self.setup_modern_ui()
        self.load_model()
    
    def setup_modern_ui(self):
        # Configure modern styling
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        
        # Main container with padding
        self.main_container = tk.Frame(self.root, bg=self.colors['bg'])
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header section
        self.create_header()
        
        # Main content area
        self.content_frame = tk.Frame(self.main_container, bg=self.colors['bg'])
        self.content_frame.pack(fill="both", expand=True, pady=(20, 0))
        
        # Create two-column layout
        self.left_panel = tk.Frame(self.content_frame, bg=self.colors['bg'])
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.right_panel = tk.Frame(self.content_frame, bg=self.colors['bg'])
        self.right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Left panel content
        self.create_controls_section()
        self.create_result_section()
        
        # Right panel content
        self.create_details_section()
        self.create_log_section()
    
    def create_header(self):
        header_frame = tk.Frame(self.main_container, bg=self.colors['bg'], height=80)
        header_frame.pack(fill="x", pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # App title with icon
        title_frame = tk.Frame(header_frame, bg=self.colors['bg'])
        title_frame.pack(side="left", fill="y")
        
        # Create a simple icon using canvas
        icon_canvas = tk.Canvas(title_frame, width=50, height=50, bg=self.colors['bg'], 
                               highlightthickness=0)
        icon_canvas.pack(side="left", padx=(0, 15))
        
        # Draw medical cross icon
        icon_canvas.create_oval(5, 5, 45, 45, fill=self.colors['primary'], outline="")
        icon_canvas.create_rectangle(20, 15, 30, 35, fill="white", outline="")
        icon_canvas.create_rectangle(15, 20, 35, 30, fill="white", outline="")
        
        title_label = tk.Label(title_frame, text="CoughDetect AI", 
                              font=("Segoe UI", 24, "bold"), 
                              bg=self.colors['bg'], fg=self.colors['text'])
        title_label.pack(side="left", anchor="center")
        
        subtitle_label = tk.Label(title_frame, text="Advanced TB Detection System", 
                                 font=("Segoe UI", 12), 
                                 bg=self.colors['bg'], fg=self.colors['text_light'])
        subtitle_label.pack(side="left", anchor="center", padx=(15, 0))
        
        # Status indicators
        status_frame = tk.Frame(header_frame, bg=self.colors['bg'])
        status_frame.pack(side="right", fill="y")
        
        self.model_status_indicator = tk.Frame(status_frame, bg=self.colors['warning'], 
                                              width=12, height=12)
        self.model_status_indicator.pack(side="right", padx=(10, 0), pady=20)
        
        self.model_status_label = tk.Label(status_frame, text="Loading Model...", 
                                          font=("Segoe UI", 10), 
                                          bg=self.colors['bg'], fg=self.colors['text_light'])
        self.model_status_label.pack(side="right", anchor="center", pady=20)
    
    def create_controls_section(self):
        controls_card = ModernCard(self.left_panel, title="Controls")
        controls_card.pack(fill="x", pady=(0, 20))
        
        controls_content = tk.Frame(controls_card, bg="white")
        controls_content.pack(fill="x", padx=20, pady=(0, 20))
        
        # File selection section
        file_section = tk.Frame(controls_content, bg="white")
        file_section.pack(fill="x", pady=(0, 20))
        
        tk.Label(file_section, text="Audio Files", font=("Segoe UI", 12, "bold"), 
                bg="white", fg=self.colors['text']).pack(anchor="w")
        
        file_buttons_frame = tk.Frame(file_section, bg="white")
        file_buttons_frame.pack(fill="x", pady=(10, 0))
        
        self.select_files_btn = ModernButton(file_buttons_frame, "📁 Select Audio Files", 
                                           command=self.select_files, width=180)
        self.select_files_btn.pack(side="left")
        
        self.file_count_label = tk.Label(file_buttons_frame, text="No files selected", 
                                        font=("Segoe UI", 10), bg="white", 
                                        fg=self.colors['text_light'])
        self.file_count_label.pack(side="left", padx=(20, 0), anchor="center")

        # --- Reset Button ---
        self.reset_btn = ModernButton(file_buttons_frame, "🔄 Reset", 
                                      command=self.reset_all, style="secondary", width=100)
        self.reset_btn.pack(side="left", padx=(20, 0))
        # --------------------

        # Recording section
        record_section = tk.Frame(controls_content, bg="white")
        record_section.pack(fill="x", pady=(0, 20))
        
        tk.Label(record_section, text="Record Coughs", font=("Segoe UI", 12, "bold"), 
                bg="white", fg=self.colors['text']).pack(anchor="w")
        
        record_buttons_frame = tk.Frame(record_section, bg="white")
        record_buttons_frame.pack(fill="x", pady=(10, 0))
        
        self.record_btn = ModernButton(record_buttons_frame, "🎤 Record Cough", 
                                     command=self.record_cough, style="success", width=150)
        self.record_btn.pack(side="left")
        
        self.cough_count_label = tk.Label(record_buttons_frame, text="Coughs recorded: 0/5", 
                                         font=("Segoe UI", 10), bg="white", 
                                         fg=self.colors['text_light'])
        self.cough_count_label.pack(side="left", padx=(20, 0), anchor="center")
        
        # Process section
        process_section = tk.Frame(controls_content, bg="white")
        process_section.pack(fill="x")
        
        tk.Label(process_section, text="Analysis", font=("Segoe UI", 12, "bold"), 
                bg="white", fg=self.colors['text']).pack(anchor="w")
        
        process_frame = tk.Frame(process_section, bg="white")
        process_frame.pack(fill="x", pady=(10, 0))
        
        self.process_btn = ModernButton(process_frame, "🔬 Analyze Coughs", 
                                      command=self.process_files, style="primary", width=160)
        self.process_btn.pack(side="left")
        self.process_btn.configure_state("disabled")
        
        # Progress bar
        self.progress_bar = AnimatedProgressBar(process_frame)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(20, 0))

    # --- Add this method to ModernTBDetectionGUI ---
    def reset_all(self):
        """Reset all selections, recordings, results, and UI for a new analysis session."""
        self.audio_files.clear()
        self.recorded_coughs.clear()
        self.results.clear()
        self.is_processing = False

        self.file_count_label.configure(text="No files selected")
        self.cough_count_label.configure(text=f"Coughs recorded: 0/{self.required_coughs}")
        self.progress_bar.set_progress(0)
        self.process_btn.configure_state("disabled")
        self.update_result_display()
        self.stats_text.delete(1.0, tk.END)
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()
        self.log_message("System reset. Ready for new analysis.", "INFO")
    
    def create_result_section(self):
        result_card = ModernCard(self.left_panel, title="Detection Result")
        result_card.pack(fill="x", pady=(0, 20))
        
        result_content = tk.Frame(result_card, bg="white", height=200)
        result_content.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        result_content.pack_propagate(False)
        
        # Large result display
        self.main_result_frame = tk.Frame(result_content, bg="white")
        self.main_result_frame.pack(fill="both", expand=True)
        
        self.result_icon_canvas = tk.Canvas(self.main_result_frame, width=80, height=80, 
                                          bg="white", highlightthickness=0)
        self.result_icon_canvas.pack(pady=(20, 10))
        
        self.main_result_label = tk.Label(self.main_result_frame, text="NO ANALYSIS PERFORMED", 
                                         font=("Segoe UI", 18, "bold"), 
                                         bg="white", fg=self.colors['text_light'])
        self.main_result_label.pack()
        self.confidence_label = tk.Label(self.main_result_frame, text="", 
                                         font=("Segoe UI", 12), 
                                         bg="white", fg=self.colors['text_light'])
        self.confidence_label.pack()
        
        # Risk level indicator
        self.risk_indicator_frame = tk.Frame(self.main_result_frame, bg="white")
        self.risk_indicator_frame.pack(pady=(10, 0))
        
        self.update_result_display()
    
    def create_details_section(self):
        details_card = ModernCard(self.right_panel, title="Analysis Details")
        details_card.pack(fill="both", expand=True, pady=(0, 20))
        
        details_content = tk.Frame(details_card, bg="white")
        details_content.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(details_content)
        self.notebook.pack(fill="both", expand=True)
        
        # Statistics tab
        self.stats_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.stats_frame, text="Statistics")
        
        # Visualization tab
        self.viz_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.viz_frame, text="Visualization")
        
        # History tab
        self.history_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.history_frame, text="History")
        
        self.setup_stats_tab()
        self.setup_viz_tab()
        self.setup_history_tab()
    
    def setup_stats_tab(self):
        stats_scroll = scrolledtext.ScrolledText(self.stats_frame, wrap=tk.WORD, 
                                               font=("Consolas", 10), height=15)
        stats_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        stats_scroll.config(bg="#f8f9fa", relief="flat", bd=0)
        self.stats_text = stats_scroll
    
    def setup_viz_tab(self):
        # Matplotlib figure for visualizations
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize empty plots
        self.ax1.set_title("Confidence Scores")
        self.ax1.set_xlabel("Sample")
        self.ax1.set_ylabel("TB Probability")
        
        self.ax2.set_title("Risk Distribution")
        self.ax2.set_xlabel("Risk Level")
        self.ax2.set_ylabel("Count")
        
        plt.tight_layout()
    
    def setup_history_tab(self):
        # History table using Treeview
        columns = ("Time", "Files", "TB Detected", "Avg Confidence", "Risk Level")
        self.history_tree = ttk.Treeview(self.history_frame, columns=columns, show="headings", height=12)
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100, anchor="center")
        
        history_scroll = ttk.Scrollbar(self.history_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        
        self.history_tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        history_scroll.pack(side="right", fill="y", pady=10, padx=(0, 10))
    
    def create_log_section(self):
        log_card = ModernCard(self.right_panel, title="System Log")
        log_card.pack(fill="x")
        
        log_content = tk.Frame(log_card, bg="white")
        log_content.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        self.log_text = scrolledtext.ScrolledText(log_content, wrap=tk.WORD, 
                                                font=("Consolas", 9), height=8)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.config(bg="#1f2937", fg="#e5e7eb", insertbackground="#6366f1",
                           relief="flat", bd=0)
    
    def log_message(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "INFO": "#3b82f6",
            "SUCCESS": "#10b981", 
            "WARNING": "#f59e0b",
            "ERROR": "#ef4444"
        }
        
        self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.log_text.insert(tk.END, f"{level}: ", level.lower())
        self.log_text.insert(tk.END, f"{message}\n")
        
        # Configure tags for colored text
        self.log_text.tag_config("timestamp", foreground="#6b7280")
        self.log_text.tag_config(level.lower(), foreground=colors.get(level, "#e5e7eb"))
        
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def load_model(self):
        """Load the ML model in a separate thread"""
        def load_model_thread():
            try:
                self.log_message("Loading TB detection model...")
                model = MobileNetV4_Res2TSM('mobilenetv4_conv_blur_medium').to(DEVICE)
                
                model_path = "final_best_mobilenetv4_conv_blur_medium_res2tsm_tb_classifier.pth"
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                state = torch.load(model_path, map_location=DEVICE)
                
                # Handle different checkpoint formats
                if 'state_dict' in state:
                    state_dict = state['state_dict']
                elif 'model_state_dict' in state:
                    state_dict = state['model_state_dict']
                else:
                    state_dict = state
                
                # Remove 'module.' prefix if present (from DataParallel)
                if any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                
                self.model = model
                self.root.after(0, lambda: [
                    self.model_status_label.configure(text="Model loaded successfully", foreground="green"),
                    self.log_message("Model loaded successfully!")
                ])
                
            except Exception as e:
                self.root.after(0, lambda: [
                    self.model_status_label.configure(text="Model load failed", foreground="red"),
                    self.log_message(f"Error loading model: {str(e)}"),
                    messagebox.showerror("Model Error", f"Failed to load model:\n{str(e)}")
                ])
        
        threading.Thread(target=load_model_thread, daemon=True).start()
    
    def on_model_loaded(self):
        self.model_status_indicator.configure(bg=self.colors['success'])
        self.model_status_label.configure(text="Model Ready")
        self.log_message("System ready for TB detection", "SUCCESS")
    
    def on_model_error(self, error):
        self.model_status_indicator.configure(bg=self.colors['danger'])
        self.model_status_label.configure(text="Model Error")
        messagebox.showerror("Model Error", f"Failed to load model:\n{error}")
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                ("WAV Files", "*.wav"),
                ("MP3 Files", "*.mp3"),
                ("All Files", "*.*")
            ]
        )
        
        if files:
            self.audio_files = list(files)
            count = len(self.audio_files)
            self.file_count_label.configure(text=f"{count} file{'s' if count != 1 else ''} selected")
            self.log_message(f"Selected {count} audio files for analysis")
            self.update_process_button_state()
    
    def record_cough(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        if len(self.recorded_coughs) >= self.required_coughs:
            response = messagebox.askyesno("Recording Limit", 
                                         f"You already have {self.required_coughs} recorded coughs. "
                                         "Do you want to clear them and start over?")
            if response:
                self.recorded_coughs.clear()
                self.update_cough_count()
            else:
                return
        
        self.is_recording = True
        self.record_btn.set_text_and_style(text="⏹️ Stop Recording", style="danger")
        self.log_message("Recording cough sample...")

        def record_audio():
            try:
                duration = 3  # seconds
                sample_rate = SR
                
                self.log_message(f"Recording for {duration} seconds...")
                audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, 
                                  channels=1, dtype=np.float32)
                sd.wait()
                
                if self.is_recording:  # Check if not cancelled
                    # Save recorded audio
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"recorded_cough_{len(self.recorded_coughs)+1}_{timestamp}.wav"
                    filepath = os.path.join(os.getcwd(), filename)
                    
                    sf.write(filepath, audio_data.flatten(), sample_rate)
                    self.recorded_coughs.append(filepath)
                    
                    self.root.after(0, self.on_recording_complete)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_recording_error(str(e)))
        
        self.recording_thread = threading.Thread(target=record_audio, daemon=True)
        self.recording_thread.start()
    
    def stop_recording(self):
        self.is_recording = False
        self.record_btn.set_text_and_style(text="🎤 Record Cough", style="success")
        self.log_message("Recording stopped")

    def on_recording_complete(self):
        self.is_recording = False
        self.record_btn.set_text_and_style(text="🎤 Record Cough", style="success")

        count = len(self.recorded_coughs)
        self.log_message(f"Cough sample {count} recorded successfully", "SUCCESS")
        self.update_cough_count()
        self.update_process_button_state()
    
    def on_recording_error(self, error):
        self.is_recording = False
        self.record_btn.set_text_and_style(text="🎤 Record Cough", style="success")
        self.log_message(f"Recording failed: {error}", "ERROR")
        messagebox.showerror("Recording Error", f"Failed to record audio:\n{error}")

    def update_cough_count(self):
        count = len(self.recorded_coughs)
        self.cough_count_label.configure(text=f"Coughs recorded: {count}/{self.required_coughs}")
    
    def update_process_button_state(self):
        has_files = len(self.audio_files) > 0 or len(self.recorded_coughs) > 0
        model_ready = self.model is not None
        
        if has_files and model_ready and not self.is_processing:
            self.process_btn.configure_state("normal")
        else:
            self.process_btn.configure_state("disabled")
    
    def process_files(self):
        if self.is_processing:
            return
        
        all_files = self.audio_files + self.recorded_coughs
        if not all_files:
            messagebox.showwarning("No Files", "Please select audio files or record coughs first.")
            return
        
        if not self.model:
            messagebox.showerror("Model Error", "Model is not loaded. Please restart the application.")
            return
        
        self.is_processing = True
        self.process_btn.configure_state("disabled")
        self.progress_bar.set_progress(0)
        
        def process_in_thread():
            try:
                results = []
                total_files = len(all_files)
                
                self.log_message(f"Starting analysis of {total_files} audio files...")
                
                for i, file_path in enumerate(all_files):
                    if not self.is_processing:  # Check for cancellation
                        break
                    
                    progress = (i / total_files) * 100
                    self.root.after(0, lambda p=progress: self.progress_bar.set_progress(p))
                    
                    self.log_message(f"Processing: {os.path.basename(file_path)}")
                    
                    # Load and process audio
                    y_seg = load_and_segment(file_path)
                    if y_seg is None:
                        self.log_message(f"Failed to load: {os.path.basename(file_path)}", "WARNING")
                        continue
                    
                    # Create mel spectrogram
                    mel_spec, mel_rgb = make_mel_rgb(y_seg)
                    if mel_rgb is None:
                        self.log_message(f"Failed to process: {os.path.basename(file_path)}", "WARNING")
                        continue
                    
                    # Model inference
                    with torch.no_grad():
                        input_tensor = torch.from_numpy(mel_rgb).float().unsqueeze(0)
                        if torch.cuda.is_available():
                            input_tensor = input_tensor.to(DEVICE)
                            self.model = self.model.to(DEVICE)
                        
                        output = self.model(input_tensor)
                        prob = output.cpu().numpy()[0]
                    
                    results.append({
                        'file': os.path.basename(file_path),
                        'full_path': file_path,
                        'probability': float(prob),
                        'prediction': 'TB Detected' if prob > 0.5 else 'No TB',
                        'confidence': f"{prob*100:.1f}%"
                    })
                    
                    self.log_message(f"Result: {results[-1]['prediction']} ({results[-1]['confidence']})")
                
                self.root.after(0, lambda r=results: self.on_processing_complete(r))
                
            except Exception as e:
                self.root.after(0, lambda e=str(e): self.on_processing_error(e))
        
        threading.Thread(target=process_in_thread, daemon=True).start()
    
    def on_processing_complete(self, results):
        self.is_processing = False
        self.results = results
        self.progress_bar.set_progress(100)
        
        if results:
            tb_detected = sum(1 for r in results if r['probability'] > 0.5)
            avg_prob = np.mean([r['probability'] for r in results])
            
            self.log_message(f"Analysis complete: {tb_detected}/{len(results)} samples indicate TB", "SUCCESS")
            
            # Update displays
            self.update_result_display(results, avg_prob)
            self.update_statistics(results)
            self.update_visualizations(results)
            self.add_to_history(results)
        else:
            self.log_message("No valid results obtained", "WARNING")
        
        self.update_process_button_state()
    
    def on_processing_error(self, error):
        self.is_processing = False
        self.progress_bar.set_progress(0)
        self.log_message(f"Processing failed: {error}", "ERROR")
        messagebox.showerror("Processing Error", f"Analysis failed:\n{error}")
        self.update_process_button_state()
    
    def update_result_display(self, results=None, avg_prob=None):
        # Clear previous display
        self.result_icon_canvas.delete("all")
        
        if results is None or len(results) == 0:
            # Default state
            self.result_icon_canvas.create_oval(15, 15, 65, 65, fill="#e5e7eb", outline="")
            self.result_icon_canvas.create_text(40, 40, text="?", font=("Arial", 24, "bold"), fill="white")
            
            self.main_result_label.configure(text="NO ANALYSIS PERFORMED", fg=self.colors['text_light'])
            self.confidence_label.configure(text="")
            
            # Clear risk indicators
            for widget in self.risk_indicator_frame.winfo_children():
                widget.destroy()
            return
        
        tb_count = sum(1 for r in results if r['probability'] > 0.5)
        total_count = len(results)
        tb_ratio = tb_count / total_count if total_count > 0 else 0
        
        # Determine overall result
        if tb_ratio >= 0.5:
            result_text = "TB DETECTED"
            result_color = self.colors['danger']
            icon_color = self.colors['danger']
            icon_symbol = "⚠"
        elif tb_ratio > 0.2:
            result_text = "POSSIBLE TB"
            result_color = self.colors['warning']
            icon_color = self.colors['warning']
            icon_symbol = "?"
        else:
            result_text = "NO TB DETECTED"
            result_color = self.colors['success']
            icon_color = self.colors['success']
            icon_symbol = "✓"
        
        # Update icon
        self.result_icon_canvas.create_oval(15, 15, 65, 65, fill=icon_color, outline="")
        self.result_icon_canvas.create_text(40, 40, text=icon_symbol, font=("Arial", 20, "bold"), fill="white")
        
        # Update labels
        self.main_result_label.configure(text=result_text, fg=result_color)
        self.confidence_label.configure(text=f"Average Confidence: {avg_prob*100:.1f}%")
        
        # Update risk indicators
        for widget in self.risk_indicator_frame.winfo_children():
            widget.destroy()
        
        risk_levels = ["Low", "Medium", "High"]
        risk_colors = [self.colors['success'], self.colors['warning'], self.colors['danger']]
        
        if tb_ratio < 0.2:
            active_risk = 0
        elif tb_ratio < 0.5:
            active_risk = 1
        else:
            active_risk = 2
        
        for i, (level, color) in enumerate(zip(risk_levels, risk_colors)):
            indicator = tk.Frame(self.risk_indicator_frame, width=60, height=6)
            indicator.pack(side="left", padx=2)
            indicator.pack_propagate(False)
            
            if i <= active_risk:
                indicator.configure(bg=color)
            else:
                indicator.configure(bg="#e5e7eb")
    
    def update_statistics(self, results):
        if not results:
            return
        
        stats_text = f"""
ANALYSIS SUMMARY
{'='*50}

Total Samples Analyzed: {len(results)}
TB Detected: {sum(1 for r in results if r['probability'] > 0.5)}
No TB: {sum(1 for r in results if r['probability'] <= 0.5)}

CONFIDENCE STATISTICS
{'='*50}

Average Probability: {np.mean([r['probability'] for r in results]):.3f}
Standard Deviation: {np.std([r['probability'] for r in results]):.3f}
Minimum: {np.min([r['probability'] for r in results]):.3f}
Maximum: {np.max([r['probability'] for r in results]):.3f}

DETAILED RESULTS
{'='*50}

"""
        
        for i, result in enumerate(results, 1):
            stats_text += f"{i:2d}. {result['file']:<30} | {result['prediction']:<12} | {result['confidence']}\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def update_visualizations(self, results):
        if not results:
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Confidence scores
        probs = [r['probability'] for r in results]
        files = [r['file'][:15] + '...' if len(r['file']) > 15 else r['file'] for r in results]
        
        colors = [self.colors['danger'] if p > 0.5 else self.colors['success'] for p in probs]
        bars = self.ax1.bar(range(len(probs)), probs, color=colors, alpha=0.7)
        
        self.ax1.set_title("TB Detection Confidence Scores")
        self.ax1.set_xlabel("Audio Sample")
        self.ax1.set_ylabel("TB Probability")
        self.ax1.set_ylim(0, 1)
        self.ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        self.ax1.set_xticks(range(len(files)))
        self.ax1.set_xticklabels(files, rotation=45, ha='right')
        self.ax1.legend()
        
        # Plot 2: Risk distribution
        risk_levels = ['Low Risk\n(0.0-0.3)', 'Medium Risk\n(0.3-0.7)', 'High Risk\n(0.7-1.0)']
        risk_counts = [
            sum(1 for p in probs if p <= 0.3),
            sum(1 for p in probs if 0.3 < p <= 0.7),
            sum(1 for p in probs if p > 0.7)
        ]
        risk_colors = [self.colors['success'], self.colors['warning'], self.colors['danger']]
        
        self.ax2.bar(risk_levels, risk_counts, color=risk_colors, alpha=0.7)
        self.ax2.set_title("Risk Level Distribution")
        self.ax2.set_ylabel("Number of Samples")
        
        plt.tight_layout()
        self.canvas.draw()
    
    def add_to_history(self, results):
        if not results:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tb_detected = sum(1 for r in results if r['probability'] > 0.5)
        avg_confidence = np.mean([r['probability'] for r in results])
        
        if avg_confidence < 0.3:
            risk_level = "Low"
        elif avg_confidence < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        self.history_tree.insert("", 0, values=(
            timestamp,
            len(results),
            f"{tb_detected}/{len(results)}",
            f"{avg_confidence:.3f}",
            risk_level
        ))

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernTBDetectionGUI(root)
    root.mainloop()
