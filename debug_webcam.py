#!/usr/bin/env python3
"""Debug script to diagnose webcam issues."""

import cv2
import sys
import platform

print("="*60)
print("WEBCAM DIAGNOSTIC")
print("="*60)

# 1. Try multiple camera indices
print("\n1. Scanning for available cameras...")
available_cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"   ✅ Camera {i}: OPEN ({int(width)}x{int(height)} @ {fps:.1f} FPS)")
        available_cameras.append(i)
        
        # Try to read a test frame
        ret, frame = cap.read()
        if ret:
            print(f"      ✓ Can read frames (shape: {frame.shape})")
        else:
            print(f"      ⚠ Can open but cannot read frames")
    else:
        print(f"   ❌ Camera {i}: FAILED")
    cap.release()

if not available_cameras:
    print("\n🔴 NO CAMERAS FOUND")
    print("\nPossible causes:")
    print("   • No physical webcam connected")
    print("   • Webcam is being used by another app (Zoom, Teams, browser)")
    print("   • Running inside WSL2 / VM / Docker without USB passthrough")
    print("   • Missing camera drivers")
    print("   • Linux: need to add user to 'video' group")
    print("\nWorkarounds:")
    print("   1. Close all apps that might use the camera")
    print("   2. If on WSL2, run the app on native Windows instead")
    print("   3. Use the video file or image loop apps instead")
else:
    print(f"\n🟢 Found {len(available_cameras)} camera(s): {available_cameras}")
    print(f"\nUse: python webcam_inference_app.py --camera {available_cameras[0]}")

# 2. Check backend info
print("\n2. OpenCV Video Backends:")
backends = [cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_AVFOUNDATION]
backend_names = {cv2.CAP_V4L2: "V4L2", cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "MSMF", cv2.CAP_AVFOUNDATION: "AVFoundation"}
for b, name in backend_names.items():
    if cv2.VideoCapture(0, b).isOpened():
        print(f"   ✅ {name} works")
    else:
        print(f"   ❌ {name} failed")
    cv2.VideoCapture(0, b).release()

# 3. Platform-specific checks
print(f"\n3. Platform: {platform.system()} {platform.release()}")
if platform.system() == "Linux":
    print("   Linux check: make sure you're in the 'video' group")
    print("   Run: sudo usermod -a -G video $USER")
    print("   Then log out and back in.")
elif platform.system() == "Windows":
    print("   Windows check: Settings > Privacy > Camera > Allow apps to access camera")
elif platform.system() == "Darwin":
    print("   Mac check: System Preferences > Security & Privacy > Camera")

print("\n" + "="*60)
