/**
 * Air Canvas Web — Premium Edition
 * Ultra-smooth drawing with neon hand skeleton visualization.
 *
 * Major improvements over v1:
 *   - Larger Bézier buffer (12 points) + 8-step interpolation
 *   - No frame throttling — runs at display refresh rate
 *   - Neon glowing hand skeleton with animated dots
 *   - Particle trail effect when drawing
 *   - Velocity-adaptive glow cursor
 *   - FPS counter
 */

import {
    HandLandmarker,
    FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// ============================================================
//  Constants
// ============================================================
const COLORS = [
    "#A855F7", "#3B82F6", "#22D3EE", "#10B981",
    "#EAB308", "#F97316", "#EF4444", "#EC4899",
    "#F0F0FF", "#64748B", "#8B5CF6", "#06B6D4",
];

const MODE = { IDLE: "idle", DRAWING: "drawing", ERASING: "erasing", SELECTING: "selecting", MOVING: "moving" };
const TIP_IDS = [4, 8, 12, 16, 20];
const IS_MOBILE = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

// Hand skeleton connection pairs (landmark index pairs)
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],       // Thumb
    [0,5],[5,6],[6,7],[7,8],       // Index
    [0,9],[9,10],[10,11],[11,12],   // Middle (via 0→9)
    [0,13],[13,14],[14,15],[15,16], // Ring (via 0→13)
    [0,17],[17,18],[18,19],[19,20], // Pinky
    [5,9],[9,13],[13,17],           // Palm cross-connections
];

// ============================================================
//  LandmarkStabilizer — Dual EMA filter (position + velocity)
// ============================================================
class LandmarkStabilizer {
    constructor(alpha = 0.5, velAlpha = 0.3) {
        this.alpha = alpha;
        this.velAlpha = velAlpha;
        this.positions = null;
        this.velocities = null;
    }

    update(raw) {
        if (!this.positions) {
            this.positions = new Float32Array(raw);
            this.velocities = new Float32Array(raw.length);
            return new Float32Array(raw);
        }

        const result = new Float32Array(raw.length);
        for (let i = 0; i < raw.length; i++) {
            const diff = raw[i] - this.positions[i];
            // Smooth velocity
            this.velocities[i] += this.velAlpha * (diff - this.velocities[i]);
            // Adaptive alpha: higher for fast movements, lower for slow
            const speed = Math.abs(this.velocities[i]);
            const adaptiveAlpha = this.alpha + (1 - this.alpha) * Math.min(speed * 8, 0.4);
            // Update position
            this.positions[i] += adaptiveAlpha * diff;
            result[i] = this.positions[i];
        }
        return result;
    }

    reset() {
        this.positions = null;
        this.velocities = null;
    }
}

// ============================================================
//  FingerDetector — distance-based (much more reliable than angles)
// ============================================================
class FingerDetector {
    constructor() {
        this.states = new Uint8Array(5);
        // Per-finger hysteresis counters to prevent flicker
        this.counters = new Int32Array(5);
        this.HYST = 3; // frames required to change state
    }

    detect(lm) {
        if (lm.length < 42) return Array.from(this.states);
        const rawStates = new Uint8Array(5);

        // === Thumb: compare tip(4) distance to wrist vs ip(3) distance to wrist ===
        const wristX = lm[0], wristY = lm[1];
        const thumbTipD = Math.hypot(lm[8] - wristX, lm[9] - wristY); // tip=4 → idx 8,9
        const thumbIpD  = Math.hypot(lm[6] - wristX, lm[7] - wristY); // ip=3  → idx 6,7
        rawStates[0] = thumbTipD > thumbIpD * 1.1 ? 1 : 0;

        // === Fingers 1-4: tip must be further from wrist than PIP joint ===
        const fingerTips = [8, 12, 16, 20]; // landmark IDs
        const fingerPips = [6, 10, 14, 18]; // PIP joint IDs
        for (let i = 0; i < 4; i++) {
            const tipIdx = fingerTips[i];
            const pipIdx = fingerPips[i];
            const tipDist = Math.hypot(lm[tipIdx*2] - wristX, lm[tipIdx*2+1] - wristY);
            const pipDist = Math.hypot(lm[pipIdx*2] - wristX, lm[pipIdx*2+1] - wristY);
            rawStates[i + 1] = tipDist > pipDist * 1.02 ? 1 : 0;
        }

        // Hysteresis: require HYST consecutive frames before switching
        for (let i = 0; i < 5; i++) {
            this.counters[i] = rawStates[i]
                ? Math.max(1, this.counters[i] + 1)
                : Math.min(-1, this.counters[i] - 1);
            if (this.counters[i] >= this.HYST && !this.states[i]) this.states[i] = 1;
            else if (this.counters[i] <= -this.HYST && this.states[i]) this.states[i] = 0;
        }
        return Array.from(this.states);
    }

    reset() { this.counters.fill(0); this.states.fill(0); }
}

// ============================================================
//  ParticleSystem — Sparkle trail when drawing
// ============================================================
class ParticleSystem {
    constructor() {
        this.particles = [];
    }

    emit(x, y, color, count = 3) {
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x, y,
                vx: (Math.random() - 0.5) * 4,
                vy: (Math.random() - 0.5) * 4,
                life: 1.0,
                decay: 0.02 + Math.random() * 0.03,
                size: 2 + Math.random() * 4,
                color,
            });
        }
        // Cap particles (lower on mobile for performance)
        const maxParticles = IS_MOBILE ? 30 : 200;
        if (this.particles.length > maxParticles) {
            this.particles.splice(0, this.particles.length - maxParticles);
        }
    }

    update() {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.vx *= 0.96;
            p.vy *= 0.96;
            p.life -= p.decay;
            if (p.life <= 0) this.particles.splice(i, 1);
        }
    }

    render(ctx) {
        for (const p of this.particles) {
            ctx.globalAlpha = p.life * 0.7;
            ctx.fillStyle = p.color;
            if (!IS_MOBILE) {
                ctx.shadowColor = p.color;
                ctx.shadowBlur = 8;
            }
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1;
        ctx.shadowBlur = 0;
    }
}

// ============================================================
//  HandRenderer — Neon skeleton + landmark dots
// ============================================================
class HandRenderer {
    constructor() {
        this.glowPhase = 0;
    }

    render(ctx, landmarks, w, h, mirrored = true) {
        if (!landmarks || landmarks.length < 42) return;

        this.glowPhase += 0.05;
        const pulseAlpha = 0.5 + 0.2 * Math.sin(this.glowPhase);

        // Compute pixel positions
        const pts = [];
        for (let i = 0; i < 21; i++) {
            const nx = mirrored ? (1 - landmarks[i*2]) : landmarks[i*2];
            const ny = landmarks[i*2+1];
            pts.push({ x: nx * w, y: ny * h });
        }

        if (IS_MOBILE) {
            // ===== MOBILE: lightweight skeleton (no gradients, no shadows) =====
            ctx.strokeStyle = `rgba(0, 229, 255, ${pulseAlpha * 0.7})`;
            ctx.lineWidth = 2;
            ctx.lineCap = "round";
            // Batch all lines into one path
            ctx.beginPath();
            for (const [a, b] of HAND_CONNECTIONS) {
                ctx.moveTo(pts[a].x, pts[a].y);
                ctx.lineTo(pts[b].x, pts[b].y);
            }
            ctx.stroke();

            // Only draw tip dots (5 instead of 21)
            ctx.fillStyle = "#00E5FF";
            for (const tipId of TIP_IDS) {
                ctx.beginPath();
                ctx.arc(pts[tipId].x, pts[tipId].y, 5, 0, Math.PI * 2);
                ctx.fill();
            }
        } else {
            // ===== DESKTOP: full neon skeleton with gradients + shadows =====
            ctx.lineWidth = 2.5;
            ctx.lineCap = "round";

            for (const [a, b] of HAND_CONNECTIONS) {
                const gradient = ctx.createLinearGradient(pts[a].x, pts[a].y, pts[b].x, pts[b].y);
                gradient.addColorStop(0, `rgba(0, 229, 255, ${pulseAlpha * 0.6})`);
                gradient.addColorStop(1, `rgba(180, 92, 255, ${pulseAlpha * 0.6})`);

                ctx.strokeStyle = gradient;
                ctx.shadowColor = "rgba(0, 229, 255, 0.4)";
                ctx.shadowBlur = 6;
                ctx.beginPath();
                ctx.moveTo(pts[a].x, pts[a].y);
                ctx.lineTo(pts[b].x, pts[b].y);
                ctx.stroke();
            }

            ctx.shadowBlur = 0;

            // Draw all 21 landmark dots with glow
            for (let i = 0; i < 21; i++) {
                const isTip = TIP_IDS.includes(i);
                const radius = isTip ? 6 : 3.5;
                const color = isTip ? "#00E5FF" : "#B45CFF";
                const glowColor = isTip ? "rgba(0, 229, 255, 0.6)" : "rgba(180, 92, 255, 0.4)";

                ctx.fillStyle = glowColor;
                ctx.shadowColor = color;
                ctx.shadowBlur = isTip ? 14 : 6;
                ctx.beginPath();
                ctx.arc(pts[i].x, pts[i].y, radius + 3, 0, Math.PI * 2);
                ctx.fill();

                ctx.shadowBlur = 0;
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(pts[i].x, pts[i].y, radius, 0, Math.PI * 2);
                ctx.fill();

                if (isTip) {
                    ctx.fillStyle = "rgba(255,255,255,0.8)";
                    ctx.beginPath();
                    ctx.arc(pts[i].x, pts[i].y, 2, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }
    }
}

// ============================================================
//  DrawingEngine — Ultra-smooth drawing
// ============================================================
class DrawingEngine {
    constructor(width, height) {
        this.width = width;
        this.height = height;

        this.offscreen = document.createElement("canvas");
        this.offscreen.width = width;
        this.offscreen.height = height;
        this.ctx = this.offscreen.getContext("2d", { willReadFrequently: false });

        this.undoStack = [];
        this.redoStack = [];
        this.maxUndo = 30;
        this.stateSaved = false;

        this.mode = MODE.IDLE;
        this.activeColor = COLORS[0];
        this.brushSize = 12;
        // Bigger eraser on mobile (screen is smaller so fingers cover more area)
        const isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
        this.eraserSize = isMobile ? 200 : 80;
        this.lastX = null;
        this.lastY = null;
        // Smoothing buffer for quadratic Bézier midpoint technique
        this.prevMidX = null;
        this.prevMidY = null;

        this.tool = "brush";
        this.selection = null;
        this.selectionPath = [];

        // Mode-switch hysteresis: require more frames on mobile (since we skip every other frame)
        this._pendingMode = MODE.IDLE;
        this._pendingCount = 0;
        this._SWITCH_FRAMES = IS_MOBILE ? 8 : 4;

        // Mobile pen-up override (toggled by on-screen button)
        this.penUp = false;
    }

    processFrame(landmarks, fingerStates, viewW, viewH) {
        const [thumb, index, middle, ring, pinky] = fingerStates;

        // If pen is lifted via button (mobile), force IDLE
        if (this.penUp) {
            if (this.mode !== MODE.IDLE) {
                this._onModeExit();
                this.mode = MODE.IDLE;
            }
            // Still return the index fingertip position so the cursor shows
            const tipId = 8;
            const normX = 1 - landmarks[tipId*2];
            const normY = landmarks[tipId*2+1];
            return { x: normX * viewW, y: normY * viewH, mode: MODE.IDLE, visible: true, penUp: true };
        }

        const tipId = 8;
        const canvasX = (1 - landmarks[tipId*2]) * this.width;
        const canvasY = landmarks[tipId*2+1] * this.height;

        let wantMode;

        const getDrawOrSelectMode = () => {
            if (this.tool === "select") {
                if (this.selection) {
                    const padding = 30;
                    if (this.mode === MODE.MOVING) return MODE.MOVING; // stay moving
                    if (canvasX >= this.selection.x - padding && canvasX <= this.selection.x + this.selection.w + padding &&
                        canvasY >= this.selection.y - padding && canvasY <= this.selection.y + this.selection.h + padding) {
                        return MODE.MOVING;
                    }
                    return MODE.SELECTING; // outside bounds, start new selection
                }
                return MODE.SELECTING;
            }
            return MODE.DRAWING;
        };

        if (IS_MOBILE) {
            // ===== MOBILE: simplified rules to prevent false idle triggers =====
            if ((index + middle + ring + pinky) >= 4) {
                wantMode = MODE.ERASING;
            } else if (index === 1) {
                wantMode = getDrawOrSelectMode();
            } else {
                wantMode = this.mode;
            }
        } else {
            // ===== DESKTOP: full gesture set =====
            if (index === 1 && middle === 1 && ring === 0 && pinky === 0) {
                wantMode = MODE.IDLE;
            } else if ((index + middle + ring + pinky) >= 4) {
                wantMode = MODE.ERASING;
            } else if (index === 1) {
                wantMode = getDrawOrSelectMode();
            } else {
                wantMode = this.mode;
            }
        }

        // Hysteresis: only switch after _SWITCH_FRAMES consecutive frames agree
        if (wantMode !== this.mode) {
            if (wantMode === this._pendingMode) {
                this._pendingCount++;
            } else {
                this._pendingMode = wantMode;
                this._pendingCount = 1;
            }
            if (this._pendingCount >= this._SWITCH_FRAMES) {
                this._onModeExit(wantMode);
                this.mode = wantMode;
                this._pendingCount = 0;
            }
        } else {
            this._pendingCount = 0;
        }

        switch (this.mode) {
            case MODE.DRAWING: return this._handleDrawing(landmarks, fingerStates, viewW, viewH);
            case MODE.ERASING: return this._handleErasing(landmarks, viewW, viewH);
            case MODE.SELECTING: return this._handleSelecting(landmarks, viewW, viewH);
            case MODE.MOVING: return this._handleMoving(landmarks, viewW, viewH);
            default: return { x: 0, y: 0, mode: MODE.IDLE, visible: false };
        }
    }

    onNoHand() {
        if (this.mode !== MODE.IDLE) { this._onModeExit(MODE.IDLE); this.mode = MODE.IDLE; }
    }

    _onModeExit(nextMode) {
        if (this.mode === MODE.SELECTING && this.selectionPath.length > 2) {
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            for (const p of this.selectionPath) {
                if (p.x < minX) minX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.x > maxX) maxX = p.x;
                if (p.y > maxY) maxY = p.y;
            }

            const w = maxX - minX;
            const h = maxY - minY;

            if (w > 10 && h > 10) {
                this._saveUndo();
                
                const path = new Path2D();
                path.moveTo(this.selectionPath[0].x, this.selectionPath[0].y);
                for (let i = 1; i < this.selectionPath.length; i++) {
                    path.lineTo(this.selectionPath[i].x, this.selectionPath[i].y);
                }
                path.closePath();

                const temp = document.createElement("canvas");
                temp.width = w;
                temp.height = h;
                const tempCtx = temp.getContext("2d");
                
                tempCtx.translate(-minX, -minY);
                tempCtx.fill(path);
                tempCtx.globalCompositeOperation = "source-in";
                tempCtx.drawImage(this.ctx.canvas, 0, 0);
                
                const imageData = tempCtx.getImageData(0, 0, w, h);

                this.ctx.save();
                this.ctx.globalCompositeOperation = "destination-out";
                this.ctx.fill(path);
                this.ctx.restore();

                this.selection = {
                    imageData: imageData,
                    x: minX, y: minY, w: w, h: h,
                    path: this.selectionPath.map(p => ({ x: p.x - minX, y: p.y - minY }))
                };
            }
            this.selectionPath = [];
        }

        if (nextMode === MODE.SELECTING && this.selection) {
            this.commitSelection();
        }

        this.lastX = null;
        this.lastY = null;
        this.prevMidX = null;
        this.prevMidY = null;
        this.stateSaved = false;
    }

    _saveUndo() {
        if (!this.stateSaved) {
            this.undoStack.push(this.ctx.getImageData(0, 0, this.width, this.height));
            if (this.undoStack.length > this.maxUndo) this.undoStack.shift();
            this.redoStack = [];
            this.stateSaved = true;
        }
    }

    _handleDrawing(lm, fs, vw, vh) {
        // Always draw from the Index finger tip (landmark 8)
        const tipId = 8; 
        const normX = 1 - lm[tipId*2];
        const normY = lm[tipId*2+1];
        const canvasX = normX * this.width;
        const canvasY = normY * this.height;

        this._saveUndo();

        this.ctx.strokeStyle = this.activeColor;
        this.ctx.lineWidth = this.brushSize;
        this.ctx.lineCap = "round";
        this.ctx.lineJoin = "round";
        this.ctx.shadowColor = this.activeColor;
        this.ctx.shadowBlur = this.brushSize * 0.4;

        if (this.lastX !== null) {
            // Quadratic Bézier midpoint smoothing for buttery curves
            const midX = (this.lastX + canvasX) / 2;
            const midY = (this.lastY + canvasY) / 2;

            if (this.prevMidX !== null) {
                this.ctx.beginPath();
                this.ctx.moveTo(this.prevMidX, this.prevMidY);
                this.ctx.quadraticCurveTo(this.lastX, this.lastY, midX, midY);
                this.ctx.stroke();
            } else {
                // First segment: simple line
                this.ctx.beginPath();
                this.ctx.moveTo(this.lastX, this.lastY);
                this.ctx.lineTo(canvasX, canvasY);
                this.ctx.stroke();
            }

            this.prevMidX = midX;
            this.prevMidY = midY;
        }

        this.ctx.shadowBlur = 0;
        this.lastX = canvasX;
        this.lastY = canvasY;

        return { x: normX * vw, y: normY * vh, mode: this.mode, visible: true, canvasX, canvasY };
    }

    _handleErasing(lm, vw, vh) {
        const wNx = 1-lm[0], wNy = lm[1];
        const mNx = 1-lm[18], mNy = lm[19];
        const palmNx = (wNx + 2*mNx)/3;
        const palmNy = (wNy + 2*mNy)/3;
        const canvasX = palmNx * this.width;
        const canvasY = palmNy * this.height;

        this._saveUndo();
        this.ctx.save();
        this.ctx.globalCompositeOperation = "destination-out";
        this.ctx.lineWidth = this.eraserSize;
        this.ctx.lineCap = "round";
        this.ctx.lineJoin = "round";

        if (this.lastX !== null) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.lastX, this.lastY);
            this.ctx.lineTo(canvasX, canvasY);
            this.ctx.stroke();
        } else {
            this.ctx.beginPath();
            this.ctx.arc(canvasX, canvasY, this.eraserSize/2, 0, Math.PI*2);
            this.ctx.fill();
        }
        this.ctx.restore();
        this.lastX = canvasX;
        this.lastY = canvasY;

        return {
            x: palmNx*vw, y: palmNy*vh,
            mode: this.mode, visible: true,
            eraserRadius: this.eraserSize/2 * vw/this.width
        };
    }

    _handleSelecting(lm, vw, vh) {
        const tipId = 8;
        const normX = 1 - lm[tipId*2];
        const normY = lm[tipId*2+1];
        const canvasX = normX * this.width;
        const canvasY = normY * this.height;

        this.selectionPath.push({ x: canvasX, y: canvasY });
        
        this.lastX = canvasX;
        this.lastY = canvasY;

        return { 
            x: normX * vw, y: normY * vh, 
            mode: this.mode, visible: true, 
            selectionPath: [...this.selectionPath]
        };
    }

    _handleMoving(lm, vw, vh) {
        const tipId = 8;
        const normX = 1 - lm[tipId*2];
        const normY = lm[tipId*2+1];
        const canvasX = normX * this.width;
        const canvasY = normY * this.height;

        if (this.selection && this.lastX !== null) {
            this.selection.x += (canvasX - this.lastX);
            this.selection.y += (canvasY - this.lastY);
        }
        this.lastX = canvasX;
        this.lastY = canvasY;

        return { 
            x: normX * vw, y: normY * vh, 
            mode: this.mode, visible: true, 
            selection: this.selection 
        };
    }

    commitSelection() {
        if (!this.selection) return;
        const temp = document.createElement("canvas");
        temp.width = this.selection.w;
        temp.height = this.selection.h;
        temp.getContext("2d").putImageData(this.selection.imageData, 0, 0);
        this.ctx.drawImage(temp, this.selection.x, this.selection.y);
        this.selection = null;
    }

    undo() {
        this.commitSelection();
        if (!this.undoStack.length) return;
        this.redoStack.push(this.ctx.getImageData(0,0,this.width,this.height));
        this.ctx.putImageData(this.undoStack.pop(), 0, 0);
    }

    redo() {
        this.commitSelection();
        if (!this.redoStack.length) return;
        this.undoStack.push(this.ctx.getImageData(0,0,this.width,this.height));
        this.ctx.putImageData(this.redoStack.pop(), 0, 0);
    }

    clearCanvas() {
        this.commitSelection();
        this._saveUndo();
        this.stateSaved = false;
        this.ctx.clearRect(0, 0, this.width, this.height);
    }

    get canUndo() { return this.undoStack.length > 0; }
    get canRedo() { return this.redoStack.length > 0; }
}

// ============================================================
//  Main App
// ============================================================
class AirCanvasApp {
    constructor() {
        this.loadingScreen = document.getElementById("loading-screen");
        this.loadingBarFill = document.getElementById("loading-bar-fill");
        this.loadingStatus = document.getElementById("loading-status");
        this.permissionScreen = document.getElementById("permission-screen");
        this.gestureGuide = document.getElementById("gesture-guide");
        this.appContainer = document.getElementById("app-container");
        this.video = document.getElementById("camera-video");
        this.drawingCanvas = document.getElementById("drawing-canvas");
        this.handCanvas = document.getElementById("hand-canvas");
        this.cursorCanvas = document.getElementById("cursor-canvas");
        this.modeIndicator = document.getElementById("mode-indicator");
        this.modeIcon = document.getElementById("mode-icon");
        this.modeText = document.getElementById("mode-text");
        this.fpsCounter = document.getElementById("fps-counter");

        this.handLandmarker = null;
        this.stabilizer = new LandmarkStabilizer(0.5, 0.3);
        this.fingerDetector = new FingerDetector();
        this.drawingEngine = null;
        this.handRenderer = new HandRenderer();
        this.particles = new ParticleSystem();

        this.preferredHand = "Right";
        this.useFrontCamera = true;
        this.activeColor = COLORS[0];
        this.brushSize = 12;
        this.isRunning = false;
        this.animationFrameId = null;
        this.lastTimestamp = 0;

        // FPS tracking
        this.fpsFrames = 0;
        this.fpsLastTime = 0;

        // Cursor trail
        this.cursorTrail = [];
        this.maxTrail = 12;

        this.drawCtx = null;
        this.handCtx = null;
        this.cursorCtx = null;

        this._setupColorPicker();
        this._setupEventListeners();
        this._init();
    }

    async _init() {
        try {
            this._setLoading(10, "Loading MediaPipe Vision...");
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
            );

            this._setLoading(45, "Creating hand landmarker...");
            this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                    delegate: "GPU",
                },
                runningMode: "VIDEO",
                numHands: 1,
                minHandDetectionConfidence: IS_MOBILE ? 0.4 : 0.5,
                minHandPresenceConfidence: IS_MOBILE ? 0.4 : 0.5,
                minTrackingConfidence: IS_MOBILE ? 0.3 : 0.5,
            });

            this._setLoading(85, "Hand tracker ready!");
            this._setLoading(100, "Starting...");

            setTimeout(() => {
                this.loadingScreen.classList.add("fade-out");
                setTimeout(() => {
                    this.loadingScreen.classList.add("hidden");
                    this._requestCamera();
                }, 600);
            }, 400);
        } catch (err) {
            console.error("Init failed:", err);
            this._setLoading(0, `Error: ${err.message}`);
        }
    }

    async _requestCamera() {
        const camWidth = IS_MOBILE ? 640 : 1280;
        const camHeight = IS_MOBILE ? 480 : 720;
        const camConfig = {
            video: { facingMode: "user", width: { ideal: camWidth }, height: { ideal: camHeight } },
            audio: false,
        };
        try {
            const stream = await navigator.mediaDevices.getUserMedia(camConfig);
            this._startWithStream(stream);
        } catch {
            this.permissionScreen.classList.remove("hidden");
            document.getElementById("grant-camera-btn").addEventListener("click", async () => {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia(camConfig);
                    this.permissionScreen.classList.add("hidden");
                    this._startWithStream(stream);
                } catch {
                    alert("Camera access denied. Please allow camera in browser settings and refresh.");
                }
            });
        }
    }

    _startWithStream(stream) {
        this.video.srcObject = stream;
        this.video.play();
        this.video.addEventListener("loadeddata", () => {
            this._setupCanvases();
            this.appContainer.classList.remove("hidden");
            this.gestureGuide.classList.remove("hidden");
            this.isRunning = true;
            this.fpsLastTime = performance.now();
            this._loop();
        }, { once: true });
    }

    _setupCanvases() {
        const w = this.video.videoWidth;
        const h = this.video.videoHeight;

        [this.drawingCanvas, this.handCanvas, this.cursorCanvas].forEach(c => {
            c.width = w;
            c.height = h;
        });

        this.drawCtx = this.drawingCanvas.getContext("2d");
        this.handCtx = this.handCanvas.getContext("2d");
        this.cursorCtx = this.cursorCanvas.getContext("2d");

        this.drawingEngine = new DrawingEngine(w, h);
        this.drawingEngine.activeColor = this.activeColor;
        this.drawingEngine.brushSize = this.brushSize;
    }

    _loop() {
        if (!this.isRunning) return;

        const now = performance.now();

        // FPS counter
        this.fpsFrames++;
        if (now - this.fpsLastTime >= 1000) {
            this.fpsCounter.textContent = `${this.fpsFrames} FPS`;
            this.fpsFrames = 0;
            this.fpsLastTime = now;
        }

        if (this.video.readyState >= 2 && now !== this.lastTimestamp) {
            // On mobile: skip detection every other frame to save CPU
            if (IS_MOBILE) {
                this._mobileFrameCount = (this._mobileFrameCount || 0) + 1;
                if (this._mobileFrameCount % 2 === 0) {
                    const results = this.handLandmarker.detectForVideo(this.video, now);
                    this._processResults(results);
                } else {
                    // Re-render with last known state (smooth interpolation)
                    this._renderDrawing();
                }
            } else {
                const results = this.handLandmarker.detectForVideo(this.video, now);
                this._processResults(results);
            }
            this.lastTimestamp = now;
        }

        this.animationFrameId = requestAnimationFrame(() => this._loop());
    }

    _processResults(results) {
        const vw = this.handCanvas.width;
        const vh = this.handCanvas.height;

        // Clear overlay canvases
        this.handCtx.clearRect(0, 0, vw, vh);
        this.cursorCtx.clearRect(0, 0, vw, vh);

        if (!results.landmarks || results.landmarks.length === 0) {
            this.stabilizer.reset();
            this.fingerDetector.reset();
            this.drawingEngine.onNoHand();
            this._updateMode(MODE.IDLE);
            this.cursorTrail = [];
            // Still render particles (fading out)
            this.particles.update();
            this.particles.render(this.cursorCtx);
            this._renderDrawing();
            return;
        }

        // Select preferred hand
        let handIdx = 0;
        if (results.handedness) {
            for (let i = 0; i < results.handedness.length; i++) {
                if (results.handedness[i][0]?.categoryName === this.preferredHand) {
                    handIdx = i;
                    break;
                }
            }
        }
        if (handIdx >= results.landmarks.length) handIdx = 0;
        const lm = results.landmarks[handIdx];

        // Flatten
        const raw = new Float32Array(lm.length * 2);
        for (let i = 0; i < lm.length; i++) {
            raw[i*2] = lm[i].x;
            raw[i*2+1] = lm[i].y;
        }

        const stabilized = this.stabilizer.update(raw);
        const fingerStates = this.fingerDetector.detect(stabilized);

        this.drawingEngine.activeColor = this.activeColor;
        this.drawingEngine.brushSize = this.brushSize;

        const cursor = this.drawingEngine.processFrame(stabilized, fingerStates, vw, vh);

        // Render hand skeleton
        this.handRenderer.render(this.handCtx, stabilized, vw, vh, true);

        // Particles
        if (cursor.mode === MODE.DRAWING && cursor.visible && cursor.canvasX != null) {
            this.particles.emit(cursor.x, cursor.y, this.activeColor, 2);
        }
        this.particles.update();
        this.particles.render(this.cursorCtx);

        // Cursor
        if (cursor.penUp) {
            this._updateMode("penup");
        } else {
            this._updateMode(cursor.mode);
        }
        this._renderDrawing();
        this._renderCursor(cursor, vw, vh);
        this._updateToolbarState();
    }

    _renderDrawing() {
        this.drawCtx.clearRect(0, 0, this.drawingCanvas.width, this.drawingCanvas.height);
        this.drawCtx.drawImage(this.drawingEngine.offscreen, 0, 0);
        
        // Render floating selection if it exists
        if (this.drawingEngine && this.drawingEngine.selection) {
            const sel = this.drawingEngine.selection;
            const temp = document.createElement("canvas");
            temp.width = sel.w;
            temp.height = sel.h;
            temp.getContext("2d").putImageData(sel.imageData, 0, 0);
            
            this.drawCtx.save();
            this.drawCtx.shadowColor = "rgba(180, 92, 255, 0.4)";
            this.drawCtx.shadowBlur = 15;
            this.drawCtx.drawImage(temp, sel.x, sel.y);
            this.drawCtx.restore();
        }
    }

    _renderCursor(cursor, vw, vh) {
        if (!cursor.visible) { this.cursorTrail = []; return; }

        const ctx = this.cursorCtx;
        const x = cursor.x, y = cursor.y;

        // Update trail
        this.cursorTrail.push({ x, y });
        if (this.cursorTrail.length > this.maxTrail) this.cursorTrail.shift();

        switch (cursor.mode) {
            case MODE.DRAWING: {
                // Draw trail
                if (this.cursorTrail.length >= 2) {
                    for (let i = 1; i < this.cursorTrail.length; i++) {
                        const alpha = i / this.cursorTrail.length * 0.3;
                        const width = i / this.cursorTrail.length * 6;
                        ctx.strokeStyle = this.activeColor;
                        ctx.globalAlpha = alpha;
                        ctx.lineWidth = width;
                        ctx.lineCap = "round";
                        ctx.beginPath();
                        ctx.moveTo(this.cursorTrail[i-1].x, this.cursorTrail[i-1].y);
                        ctx.lineTo(this.cursorTrail[i].x, this.cursorTrail[i].y);
                        ctx.stroke();
                    }
                    ctx.globalAlpha = 1;
                }

                // Outer glow
                const grad = ctx.createRadialGradient(x, y, 0, x, y, 28);
                grad.addColorStop(0, this.activeColor + "66");
                grad.addColorStop(0.5, this.activeColor + "22");
                grad.addColorStop(1, this.activeColor + "00");
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(x, y, 28, 0, Math.PI * 2);
                ctx.fill();

                // Inner dot
                ctx.fillStyle = this.activeColor;
                ctx.shadowColor = this.activeColor;
                ctx.shadowBlur = 16;
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fill();

                // White center
                ctx.shadowBlur = 0;
                ctx.fillStyle = "rgba(255,255,255,0.9)";
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
                break;
            }
            case MODE.ERASING: {
                const r = cursor.eraserRadius || 40;
                // Animated dashed circle
                ctx.setLineDash([8, 6]);
                ctx.strokeStyle = "rgba(255,100,100,0.5)";
                ctx.lineWidth = 2.5;
                ctx.shadowColor = "rgba(255,100,100,0.3)";
                ctx.shadowBlur = 10;
                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.shadowBlur = 0;

                // Center X
                ctx.strokeStyle = "rgba(255,100,100,0.4)";
                ctx.lineWidth = 2;
                const s = 8;
                ctx.beginPath();
                ctx.moveTo(x-s, y-s); ctx.lineTo(x+s, y+s);
                ctx.moveTo(x+s, y-s); ctx.lineTo(x-s, y+s);
                ctx.stroke();
                break;
            }
            case MODE.SELECTING: {
                if (cursor.selectionPath && cursor.selectionPath.length > 0) {
                    const scaleX = vw / this.drawingEngine.width;
                    const scaleY = vh / this.drawingEngine.height;
                    
                    ctx.setLineDash([8, 8]);
                    ctx.strokeStyle = "#00E5FF";
                    ctx.lineWidth = 2;
                    ctx.shadowColor = "#00E5FF";
                    ctx.shadowBlur = 8;
                    
                    ctx.beginPath();
                    ctx.moveTo(cursor.selectionPath[0].x * scaleX, cursor.selectionPath[0].y * scaleY);
                    for (let i = 1; i < cursor.selectionPath.length; i++) {
                        ctx.lineTo(cursor.selectionPath[i].x * scaleX, cursor.selectionPath[i].y * scaleY);
                    }
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.shadowBlur = 0;
                }
                ctx.strokeStyle = "#00E5FF";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x-8, y); ctx.lineTo(x+8, y);
                ctx.moveTo(x, y-8); ctx.lineTo(x, y+8);
                ctx.stroke();
                break;
            }
            case MODE.MOVING: {
                if (cursor.selection) {
                    const sel = cursor.selection;
                    const sx = sel.x * (vw / this.drawingEngine.width);
                    const sy = sel.y * (vh / this.drawingEngine.height);
                    const sw = sel.w * (vw / this.drawingEngine.width);
                    const sh = sel.h * (vh / this.drawingEngine.height);
                    const scaleX = vw / this.drawingEngine.width;
                    const scaleY = vh / this.drawingEngine.height;
                    
                    if (sel.path && sel.path.length > 0) {
                        ctx.setLineDash([6, 6]);
                        ctx.strokeStyle = "#B45CFF";
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(sx + sel.path[0].x * scaleX, sy + sel.path[0].y * scaleY);
                        for (let i = 1; i < sel.path.length; i++) {
                            ctx.lineTo(sx + sel.path[i].x * scaleX, sy + sel.path[i].y * scaleY);
                        }
                        ctx.closePath();
                        ctx.stroke();
                        ctx.setLineDash([]);
                    } else {
                        ctx.setLineDash([6, 6]);
                        ctx.strokeStyle = "#B45CFF";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(sx, sy, sw, sh);
                        ctx.setLineDash([]);
                    }
                }
                ctx.fillStyle = "#B45CFF";
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fill();
                break;
            }
        }

        // Ghost cursor when pen is up (draw after switch so it overlays)
        if (cursor.penUp) {
            // Dashed circle
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = "rgba(249, 115, 22, 0.5)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, 18, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // Inner dimmed dot
            ctx.globalAlpha = 0.4;
            ctx.fillStyle = "#F97316";
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1;
        }

    }

    _updateMode(mode) {
        this.modeIndicator.setAttribute("data-mode", mode);
        const cfg = {
            [MODE.IDLE]: { icon: "✊", text: "Idle" },
            [MODE.DRAWING]: { icon: "✏️", text: "Drawing" },
            [MODE.ERASING]: { icon: "🧹", text: "Erasing" },
            [MODE.SELECTING]: { icon: "✂️", text: "Selecting" },
            [MODE.MOVING]: { icon: "✋", text: "Moving" },
            "penup": { icon: "✋", text: "Pen Up" },
        };
        const c = cfg[mode] || cfg[MODE.IDLE];
        this.modeIcon.textContent = c.icon;
        this.modeText.textContent = c.text;
    }

    _updateToolbarState() {
        document.getElementById("undo-btn").disabled = !this.drawingEngine.canUndo;
        document.getElementById("redo-btn").disabled = !this.drawingEngine.canRedo;
    }

    _setupColorPicker() {
        const grid = document.getElementById("color-grid");
        COLORS.forEach((color, i) => {
            const swatch = document.createElement("div");
            swatch.className = "color-swatch" + (i === 0 ? " active" : "");
            swatch.style.backgroundColor = color;
            swatch.style.color = color;
            swatch.addEventListener("click", () => {
                grid.querySelectorAll(".color-swatch").forEach(s => s.classList.remove("active"));
                swatch.classList.add("active");
                this.activeColor = color;
                document.getElementById("color-indicator").style.background = color;
                document.getElementById("color-indicator").style.boxShadow = `0 0 12px ${color}`;
                document.getElementById("brush-preview").style.background = color;
                document.getElementById("brush-preview").style.boxShadow = `0 0 12px ${color}`;
                document.getElementById("color-picker").classList.add("hidden");
            });
            grid.appendChild(swatch);
        });
    }

    _setupEventListeners() {
        document.getElementById("guide-dismiss-btn").addEventListener("click", () => {
            this.gestureGuide.classList.add("hidden");
        });

        // Mobile pen up/down toggle
        const penToggle = document.getElementById("pen-toggle-btn");
        if (penToggle) {
            penToggle.addEventListener("click", () => {
                if (!this.drawingEngine) return;
                this.drawingEngine.penUp = !this.drawingEngine.penUp;
                const icon = document.getElementById("pen-toggle-icon");
                const label = document.getElementById("pen-toggle-label");
                if (this.drawingEngine.penUp) {
                    icon.textContent = "✋";
                    label.textContent = "Pen Up";
                    penToggle.classList.add("pen-up");
                } else {
                    icon.textContent = "✏️";
                    label.textContent = "Pen Down";
                    penToggle.classList.remove("pen-up");
                }
            });
        }

        document.getElementById("help-btn").addEventListener("click", () => {
            this.gestureGuide.classList.toggle("hidden");
        });

        document.getElementById("color-btn").addEventListener("click", () => {
            if (this.drawingEngine) {
                this.drawingEngine.commitSelection();
                this.drawingEngine.tool = "brush";
            }
            document.getElementById("color-picker").classList.toggle("hidden");
            document.getElementById("brush-panel").classList.add("hidden");
        });

        document.getElementById("brush-btn").addEventListener("click", () => {
            if (this.drawingEngine) {
                this.drawingEngine.commitSelection();
                this.drawingEngine.tool = "brush";
            }
            document.getElementById("brush-panel").classList.toggle("hidden");
            document.getElementById("color-picker").classList.add("hidden");
        });

        const selectBtn = document.getElementById("select-btn");
        if (selectBtn) {
            selectBtn.addEventListener("click", () => {
                if (this.drawingEngine) {
                    this.drawingEngine.commitSelection();
                    this.drawingEngine.tool = "select";
                }
                document.getElementById("brush-panel").classList.add("hidden");
                document.getElementById("color-picker").classList.add("hidden");
                this._showToast("Selector Pen activated ✂️");
            });
        }

        const slider = document.getElementById("brush-slider");
        const sizeLabel = document.getElementById("brush-size-label");
        const preview = document.getElementById("brush-preview");
        slider.addEventListener("input", () => {
            this.brushSize = parseInt(slider.value);
            sizeLabel.textContent = this.brushSize;
            const s = Math.min(this.brushSize, 40);
            preview.style.width = s + "px";
            preview.style.height = s + "px";
        });

        document.getElementById("undo-btn").addEventListener("click", () => {
            this.drawingEngine?.undo();
            this._renderDrawing();
            this._updateToolbarState();
        });

        document.getElementById("redo-btn").addEventListener("click", () => {
            this.drawingEngine?.redo();
            this._renderDrawing();
            this._updateToolbarState();
        });

        document.getElementById("clear-btn").addEventListener("click", () => {
            this.drawingEngine?.clearCanvas();
            this._renderDrawing();
            this._updateToolbarState();
            this._showToast("Canvas cleared 🧹");
        });

        document.getElementById("save-btn").addEventListener("click", () => {
            if (!this.drawingEngine) return;
            
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = this.drawingEngine.width;
            tempCanvas.height = this.drawingEngine.height;
            const tCtx = tempCanvas.getContext("2d");
            
            // Solid dark background for neon contrast
            tCtx.fillStyle = "#050505";
            tCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
            
            // Draw artwork
            tCtx.drawImage(this.drawingEngine.offscreen, 0, 0);
            
            // Authenticity Watermark
            tCtx.font = "bold 24px sans-serif";
            tCtx.fillStyle = "rgba(255, 255, 255, 0.6)";
            tCtx.textAlign = "right";
            tCtx.fillText("✨ Tarun Panthi's Air Canvas", tempCanvas.width - 24, tempCanvas.height - 24);

            const link = document.createElement("a");
            link.download = `TarunPanthi_AirCanvas_${Date.now()}.png`;
            link.href = tempCanvas.toDataURL("image/png");
            link.click();
            this._showToast("Authentic Artwork saved! 🎉");
        });

        document.getElementById("hand-toggle-btn").addEventListener("click", () => {
            this.preferredHand = this.preferredHand === "Right" ? "Left" : "Right";
            document.getElementById("hand-label").textContent = this.preferredHand;
            this._showToast(`Switched to ${this.preferredHand} hand`);
        });

        document.getElementById("camera-flip-btn").addEventListener("click", async () => {
            this.useFrontCamera = !this.useFrontCamera;
            this.isRunning = false;
            cancelAnimationFrame(this.animationFrameId);
            if (this.video.srcObject) {
                this.video.srcObject.getTracks().forEach(t => t.stop());
            }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: this.useFrontCamera ? "user" : "environment",
                        width: { ideal: 1280 }, height: { ideal: 720 },
                    },
                    audio: false,
                });
                this.video.style.transform = this.useFrontCamera ? "scaleX(-1)" : "scaleX(1)";
                this._startWithStream(stream);
            } catch {
                this._showToast("Failed to switch camera");
                this.useFrontCamera = !this.useFrontCamera;
            }
        });

        window.addEventListener("beforeunload", () => {
            if (this.video && this.video.srcObject) {
                this.video.srcObject.getTracks().forEach(t => t.stop());
            }
        });

        document.addEventListener("keydown", (e) => {
            if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
                if (e.shiftKey) this.drawingEngine?.redo();
                else this.drawingEngine?.undo();
                this._renderDrawing();
                this._updateToolbarState();
                e.preventDefault();
            }
            if (e.key === "c" && !e.ctrlKey && !e.metaKey) {
                this.drawingEngine?.clearCanvas();
                this._renderDrawing();
                this._updateToolbarState();
            }
            if (e.key === "s" && (e.ctrlKey || e.metaKey)) {
                document.getElementById("save-btn").click();
                e.preventDefault();
            }
        });
    }

    _showToast(message) {
        document.querySelectorAll(".toast").forEach(t => t.remove());
        const toast = document.createElement("div");
        toast.className = "toast";
        toast.textContent = message;
        document.body.appendChild(toast);
        requestAnimationFrame(() => {
            toast.classList.add("show");
            setTimeout(() => {
                toast.classList.remove("show");
                setTimeout(() => toast.remove(), 500);
            }, 2000);
        });
    }

    _setLoading(percent, status) {
        this.loadingBarFill.style.width = percent + "%";
        this.loadingStatus.textContent = status;
    }
}

// ---------- Launch ----------
const app = new AirCanvasApp();
