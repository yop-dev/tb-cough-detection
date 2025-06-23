import os
import sys
import subprocess
import shutil
import platform

def build_executable():
    """Build an executable version of the TB Cough Detection application"""
    print("Building TB Cough Detection System executable...")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Create spec file with all necessary options
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['inference.py'],
    pathex=[],
    binaries=[],
    datas=[('final_best_mobilenetv4_conv_blur_medium_res2tsm_tb_classifier.pth', '.')],
    hiddenimports=['timm', 'librosa', 'sounddevice', 'soundfile', 'matplotlib', 'PIL', 'scipy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TB_Cough_Detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)
"""
    
    # Create a simple icon for the application
    create_icon()
    
    # Write the spec file
    with open("tb_detection.spec", "w") as f:
        f.write(spec_content)
    
    # Build the executable
    print("Building executable with PyInstaller...")
    subprocess.check_call([
        sys.executable, 
        "-m", 
        "PyInstaller", 
        "tb_detection.spec", 
        "--clean"
    ])
    
    # Check if build was successful
    dist_dir = os.path.join(os.getcwd(), "dist", "TB_Cough_Detection")
    if platform.system() == "Windows":
        dist_dir += ".exe"
    
    if os.path.exists(dist_dir):
        print(f"Build successful! Executable created at: {dist_dir}")
    else:
        print("Build failed. Check the PyInstaller output for errors.")

def create_icon():
    """Create a simple icon for the application"""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
        
        # Create a simple icon
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        
        # Draw a simple icon - blue circle with red center
        ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='#007bff'))
        ax.add_patch(plt.Circle((0.5, 0.5), 0.2, color='#dc3545'))
        
        # Add text
        ax.text(0.5, 0.5, 'TB', ha='center', va='center', color='white', 
                fontsize=20, fontweight='bold')
        
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        # Save as PNG first
        plt.savefig('icon.png', transparent=False)
        plt.close(fig)
        
        # Convert to ICO
        img = Image.open('icon.png')
        img.save('icon.ico')
        
        # Clean up PNG
        os.remove('icon.png')
        
        print("Application icon created successfully.")
    except Exception as e:
        print(f"Could not create icon: {e}")
        print("Continuing without custom icon...")

if __name__ == "__main__":
    build_executable()