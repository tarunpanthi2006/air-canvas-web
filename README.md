# Air Canvas Web 🎨

**Draw in the air using hand gestures — right in your browser!**

Air Canvas uses AI-powered hand tracking (MediaPipe) to let you draw, erase, and create art using just your fingers. No downloads required — works on any device with a camera.

## 🚀 Try It Live

**[Open Air Canvas →](https://YOUR_USERNAME.github.io/air-canvas-web/)**

> Replace `YOUR_USERNAME` with your GitHub username after deployment.

## ✨ Features

- **🖐️ Hand Gesture Drawing** — Point one finger to draw, open palm to erase
- **🎨 12 Color Palette** — Vibrant colors from purple to teal
- **🖌️ Adjustable Brush** — 3px to 50px brush sizes
- **↩️ Undo/Redo** — Full 30-step history
- **💾 Save Artwork** — Download your art as PNG
- **📱 Works Everywhere** — Desktop & mobile browsers
- **🔒 100% Private** — All processing happens locally, nothing leaves your device

## 🎮 Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝️ 1 Finger | Draw with active color |
| ✌️ 2-3 Fingers | Move / Navigate |
| 🖐️ Open Palm | Erase |
| ✊ Fist | Pause (idle) |

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/⌘ + Z` | Undo |
| `Ctrl/⌘ + Shift + Z` | Redo |
| `C` | Clear canvas |
| `Ctrl/⌘ + S` | Save artwork |

## 🛠️ Deploy to GitHub Pages (Free)

1. **Create a new GitHub repository** named `air-canvas-web`
2. **Push this folder**:
   ```bash
   cd AirCanvasWeb
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/air-canvas-web.git
   git push -u origin main
   ```
3. **Enable GitHub Pages**:
   - Go to your repo → **Settings** → **Pages**
   - Source: **GitHub Actions**
   - It will auto-deploy using the included workflow

4. Your site will be live at `https://YOUR_USERNAME.github.io/air-canvas-web/`

## 🏗️ Tech Stack

- **MediaPipe Tasks Vision** — Hand landmark detection (21 points per hand)
- **Canvas 2D API** — Smooth drawing with Bézier interpolation
- **WebRTC** — Camera access via `getUserMedia`
- **Vanilla HTML/CSS/JS** — Zero dependencies, zero build step

## 📁 Project Structure

```
AirCanvasWeb/
├── index.html      # Main HTML page
├── style.css       # Dark glassmorphism theme
├── app.js          # Full app: MediaPipe + drawing engine + UI
└── README.md       # This file
```

## 📄 License

MIT — Use it, modify it, share it.
