"use client";
import React, { useEffect, useRef, useState } from "react";
import { playChordFromPitchClass, playNoteFromPitchClass, playIntervalFromPitchClasses, playNotesFromPitchClasses, resumeAudioIfNeeded } from "./audio";

// Simple Tonnetz-like infinite lattice visualization.
// Each lattice node maps to a pitch class: pitch = (u*7 + v*4) mod 12
// (we use two basis vectors that correspond roughly to fifths and major thirds).

const PITCH_NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];

function pitchClassForCoords(u: number, v: number) {
    // using 7 and 4 as axis intervals gives a usable Tonnetz mapping
    const pc = ((u * 7 + v * 4) % 12 + 12) % 12;
    return pc;
}

export default function Tonnetz() {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const rafRef = useRef<number | null>(null);

    // world transform
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const offsetRef = useRef(offset);
    offsetRef.current = offset;
    const [scale, setScale] = useState(1);
    const scaleRef = useRef(scale);
    scaleRef.current = scale;

    // interaction
    const dragRef = useRef<{ down: boolean; sx: number; sy: number } | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext('2d')!;

        function resize() {
            const dpr = window.devicePixelRatio || 1;
            canvas.width = Math.floor(canvas.clientWidth * dpr);
            canvas.height = Math.floor(canvas.clientHeight * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        resize();
        window.addEventListener('resize', resize);

        function draw() {
            const w = canvas.clientWidth;
            const h = canvas.clientHeight;
            ctx.clearRect(0, 0, w, h);

            // transform world->screen
            const s = scaleRef.current;
            const ox = offsetRef.current.x;
            const oy = offsetRef.current.y;

            // lattice geometry
            const spacing = 64 * s; // pixels per lattice step
            const angle = Math.PI / 3; // 60deg
            const a1 = { x: spacing, y: 0 };
            const a2 = { x: spacing * Math.cos(angle), y: spacing * Math.sin(angle) };

            // determine visible bounds in lattice coords
            // invert transform: screen -> worldCoord = (p - offset) expressed in basis a1,a2
            function screenToWorld(px: number, py: number) {
                const x = px - w / 2 - ox;
                const y = py - h / 2 - oy;
                // solve for u,v in x = u*a1.x + v*a2.x, y = u*a1.y + v*a2.y
                const det = a1.x * a2.y - a2.x * a1.y;
                const u = (x * a2.y - a2.x * y) / det;
                const v = (a1.x * y - x * a1.y) / det;
                return { u, v };
            }

            const corners = [
                screenToWorld(0, 0),
                screenToWorld(w, 0),
                screenToWorld(0, h),
                screenToWorld(w, h),
            ];
            const us = corners.map(c => c.u);
            const vs = corners.map(c => c.v);
            const umin = Math.floor(Math.min(...us)) - 2;
            const umax = Math.ceil(Math.max(...us)) + 2;
            const vmin = Math.floor(Math.min(...vs)) - 2;
            const vmax = Math.ceil(Math.max(...vs)) + 2;

            // draw edges (triangles) with clearer styling per axis
            ctx.lineCap = 'round';
            for (let u = umin; u <= umax; u++) {
                for (let v = vmin; v <= vmax; v++) {
                    const x = w / 2 + ox + u * a1.x + v * a2.x;
                    const y = h / 2 + oy + u * a1.y + v * a2.y;
                    // draw connections to neighbours (three principal directions)
                    [[1, 0], [0, 1], [1, -1]].forEach(([du, dv]) => {
                        const x2 = w / 2 + ox + (u + du) * a1.x + (v + dv) * a2.x;
                        const y2 = h / 2 + oy + (u + du) * a1.y + (v + dv) * a2.y;
                        ctx.beginPath();
                        ctx.moveTo(x, y);
                        ctx.lineTo(x2, y2);
                        // color / width by direction so edges are visible and informative
                        const dir = `${du},${dv}`;
                        // color choices: fifths (1,0)=green, major third (0,1)=indigo, diagonal (1,-1)=orange
                        if (dir === '1,0') ctx.strokeStyle = 'rgba(34,197,94,0.9)';
                        else if (dir === '0,1') ctx.strokeStyle = 'rgba(99,102,241,0.9)';
                        else ctx.strokeStyle = 'rgba(249,115,22,0.9)';
                        ctx.lineWidth = Math.max(1, 1.2 * s);
                        ctx.stroke();
                    });
                }
            }

            // draw nodes
            for (let u = umin; u <= umax; u++) {
                for (let v = vmin; v <= vmax; v++) {
                    const x = w / 2 + ox + u * a1.x + v * a2.x;
                    const y = h / 2 + oy + u * a1.y + v * a2.y;
                    const pc = pitchClassForCoords(u, v);
                    const hue = (pc / 12) * 360;
                    ctx.fillStyle = `hsl(${hue}deg 70% 45%)`;
                    ctx.beginPath();
                    ctx.arc(x, y, Math.max(6, 8 * s), 0, Math.PI * 2);
                    ctx.fill();
                    // label
                    ctx.font = `${8 * Math.max(1, s)}px sans-serif`;
                    ctx.fillStyle = 'rgba(255,255,255,0.98)';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(PITCH_NAMES[pc], x, y);
                }
            }
        }

        const loop = () => {
            draw();
            rafRef.current = requestAnimationFrame(loop);
        };
        rafRef.current = requestAnimationFrame(loop);

        return () => {
            window.removeEventListener('resize', resize);
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, []);

    // pointer handlers for pan and click
    useEffect(() => {
        const canvas = canvasRef.current!;

        const onPointerDown = (ev: PointerEvent) => {
            (ev.target as Element).setPointerCapture(ev.pointerId);
            dragRef.current = { down: true, sx: ev.clientX, sy: ev.clientY };
        };

        const onPointerMove = (ev: PointerEvent) => {
            if (!dragRef.current?.down) return;
            const dx = ev.clientX - dragRef.current.sx;
            const dy = ev.clientY - dragRef.current.sy;
            dragRef.current.sx = ev.clientX;
            dragRef.current.sy = ev.clientY;
            setOffset(o => ({ x: o.x + dx, y: o.y + dy }));
        };

        const onPointerUp = async (ev: PointerEvent) => {
            if (!dragRef.current) return;
            const moved = Math.hypot(ev.clientX - dragRef.current.sx, ev.clientY - dragRef.current.sy);
            dragRef.current.down = false;
            // If it was a short click (not a drag), treat as click
            if (moved < 6) {
                await resumeAudioIfNeeded();
                handleClick(ev.clientX, ev.clientY);
            }
        };

        function handleClick(cx: number, cy: number) {
            const canvasRect = canvas.getBoundingClientRect();
            const px = cx - canvasRect.left;
            const py = cy - canvasRect.top;
            const w = canvas.clientWidth;
            const h = canvas.clientHeight;
            const s = scaleRef.current;
            const ox = offsetRef.current.x;
            const oy = offsetRef.current.y;
            const spacing = 64 * s;
            const angle = Math.PI / 3;
            const a1 = { x: spacing, y: 0 };
            const a2 = { x: spacing * Math.cos(angle), y: spacing * Math.sin(angle) };

            // world coords (fractional lattice coords)
            const x = px - w / 2 - ox;
            const y = py - h / 2 - oy;
            const det = a1.x * a2.y - a2.x * a1.y;
            const uFloat = (x * a2.y - a2.x * y) / det;
            const vFloat = (a1.x * y - x * a1.y) / det;

            // compute nearest lattice node and distances in screen space
            const nodeScreenRadius = Math.max(6, 8 * s);
            const clickPoint = { x: px, y: py };

            // helper: map lattice coords -> screen coords
            const latticeToScreen = (u: number, v: number) => ({
                x: w / 2 + ox + u * a1.x + v * a2.x,
                y: h / 2 + oy + u * a1.y + v * a2.y,
            });

            // find containing triangle in uv unit cell
            const iu = Math.floor(uFloat);
            const iv = Math.floor(vFloat);
            const fu = uFloat - iu;
            const fv = vFloat - iv;

            let triVerts: { u: number; v: number }[];
            if (fu + fv < 1) {
                triVerts = [{ u: iu, v: iv }, { u: iu + 1, v: iv }, { u: iu, v: iv + 1 }];
            } else {
                triVerts = [{ u: iu + 1, v: iv + 1 }, { u: iu + 1, v: iv }, { u: iu, v: iv + 1 }];
            }

            // distances to triangle vertices
            const vertScreens = triVerts.map(p => latticeToScreen(p.u, p.v));
            const vertDists = vertScreens.map(ps => Math.hypot(ps.x - clickPoint.x, ps.y - clickPoint.y));
            const minVertIdx = vertDists.indexOf(Math.min(...vertDists));

            // if we clicked sufficiently close to a node -> single note
            const nodeHitThreshold = nodeScreenRadius + 6; // a little padding
            if (vertDists[minVertIdx] <= nodeHitThreshold) {
                const vtx = triVerts[minVertIdx];
                const pc = pitchClassForCoords(vtx.u, vtx.v);
                playNoteFromPitchClass(pc, { duration: 1.6 });
                return;
            }

            // helper: distance from point to segment
            const pointToSegmentDist = (p: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) => {
                const vx = b.x - a.x;
                const vy = b.y - a.y;
                const wx = p.x - a.x;
                const wy = p.y - a.y;
                const c1 = vx * wx + vy * wy;
                if (c1 <= 0) return Math.hypot(p.x - a.x, p.y - a.y);
                const c2 = vx * vx + vy * vy;
                if (c2 <= c1) return Math.hypot(p.x - b.x, p.y - b.y);
                const t = c1 / c2;
                const projx = a.x + t * vx;
                const projy = a.y + t * vy;
                return Math.hypot(p.x - projx, p.y - projy);
            };

            // check edges of the triangle for proximity
            const edgeThreshold = 10 * Math.max(1, s);
            const edges: Array<{ aIdx: number; bIdx: number; dist: number }> = [];
            const edgePairs: [number, number][] = [[0, 1], [1, 2], [2, 0]];
            for (const [ai, bi] of edgePairs) {
                const d = pointToSegmentDist(clickPoint, vertScreens[ai], vertScreens[bi]);
                edges.push({ aIdx: ai, bIdx: bi, dist: d });
            }
            edges.sort((a, b) => a.dist - b.dist);
            if (edges[0].dist <= edgeThreshold) {
                // play the two endpoints of the nearest edge as an interval
                const aV = triVerts[edges[0].aIdx];
                const bV = triVerts[edges[0].bIdx];
                const pcA = pitchClassForCoords(aV.u, aV.v);
                const pcB = pitchClassForCoords(bV.u, bV.v);
                playIntervalFromPitchClasses(pcA, pcB, { duration: 1.6 });
                return;
            }

            // otherwise treat as inside triangle -> play the three vertices together
            const pcs = triVerts.map(v => pitchClassForCoords(v.u, v.v));
            // if the three pcs happen to form a major triad with a root we could use playChordFromPitchClass,
            // but to be general we play the three pitch-classes together
            playNotesFromPitchClasses(pcs, { duration: 1.6 });
        }

        const onWheel = (ev: WheelEvent) => {
            ev.preventDefault();
            const delta = -ev.deltaY;
            const factor = delta > 0 ? 1.08 : 0.92;
            // zoom around cursor
            const rect = canvas.getBoundingClientRect();
            const cx = ev.clientX - rect.left - rect.width / 2 - offsetRef.current.x;
            const cy = ev.clientY - rect.top - rect.height / 2 - offsetRef.current.y;
            const newScale = Math.min(4, Math.max(0.2, scaleRef.current * factor));
            // adjust offset so the point under cursor stays under cursor
            const ratio = newScale / scaleRef.current;
            setScale(newScale);
            setOffset(o => ({ x: o.x - cx * (ratio - 1), y: o.y - cy * (ratio - 1) }));
        };

        canvas.addEventListener('pointerdown', onPointerDown);
        window.addEventListener('pointermove', onPointerMove);
        window.addEventListener('pointerup', onPointerUp);
        canvas.addEventListener('wheel', onWheel, { passive: false });

        return () => {
            canvas.removeEventListener('pointerdown', onPointerDown);
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('pointerup', onPointerUp);
            canvas.removeEventListener('wheel', onWheel);
        };
    }, []);

    // sync refs when state changes
    useEffect(() => { offsetRef.current = offset; }, [offset]);
    useEffect(() => { scaleRef.current = scale; }, [scale]);

    return (
        <div style={{ position: 'fixed', inset: 0, width: '100%', height: '100%', overflow: 'hidden' }}>
            <canvas
                ref={canvasRef}
                style={{ width: '100%', height: '100%', display: 'block', touchAction: 'none', cursor: 'grab' }}
            />
            <div style={{ position: 'fixed', left: 12, top: 12, background: 'rgba(0,0,0,0.6)', color: 'white', padding: '6px 8px', borderRadius: 6, fontSize: 12, zIndex: 1000 }}>
                Click a node to play a triad. Drag to pan. Scroll to zoom.
            </div>
        </div>
    );
}
