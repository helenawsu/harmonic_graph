"use client";
import React, { useRef, useEffect, useState } from "react";

interface TensionCurveEditorProps {
  values: number[];
  onChange: (values: number[]) => void;
  minValue?: number;
  maxValue?: number;
}

export default function TensionCurveEditor({
  values,
  onChange,
  minValue = -0.15,
  maxValue = 1,
}: TensionCurveEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);

  const PADDING = 20;
  const POINT_RADIUS = 10;
  const WIDTH = 400;
  const HEIGHT = 200;

  const valueToY = (value: number) => {
    const range = maxValue - minValue;
    const normalized = (value - minValue) / range;
    return PADDING + (1 - normalized) * (HEIGHT - 2 * PADDING);
  };

  const yToValue = (y: number) => {
    const range = maxValue - minValue;
    const normalized = 1 - (y - PADDING) / (HEIGHT - 2 * PADDING);
    return Math.max(minValue, Math.min(maxValue, normalized * range + minValue));
  };

  const indexToX = (index: number) => {
    const usableWidth = WIDTH - 2 * PADDING;
    return PADDING + (index / (values.length - 1)) * usableWidth;
  };

  const findPointAtPosition = (x: number, y: number): number | null => {
    for (let i = 0; i < values.length; i++) {
      const px = indexToX(i);
      const py = valueToY(values[i]);
      const dist = Math.sqrt((x - px) ** 2 + (y - py) ** 2);
      if (dist <= POINT_RADIUS + 5) {
        return i;
      }
    }
    return null;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#222";
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    const zeroY = valueToY(0);
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PADDING, zeroY);
    ctx.lineTo(WIDTH - PADDING, zeroY);
    ctx.stroke();

    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = indexToX(i);
      const y = valueToY(values[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    for (let i = 0; i < values.length; i++) {
      const x = indexToX(i);
      const y = valueToY(values[i]);
      ctx.beginPath();
      ctx.arc(x, y, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = "#000";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.fillStyle = "#888";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    for (let i = 0; i < values.length; i++) {
      ctx.fillText(String(i + 1), indexToX(i), HEIGHT - 5);
    }
  }, [values, draggingIndex, minValue, maxValue]);

  const getMousePos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = WIDTH / rect.width;
    const scaleY = HEIGHT / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = getMousePos(e);
    const pointIndex = findPointAtPosition(x, y);
    if (pointIndex !== null) {
      setDraggingIndex(pointIndex);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (draggingIndex === null) return;
    const { y } = getMousePos(e);
    const newValue = yToValue(y);
    const newValues = [...values];
    newValues[draggingIndex] = Math.round(newValue * 100) / 100;
    onChange(newValues);
  };

  const handleMouseUp = () => {
    setDraggingIndex(null);
  };

  return (
    <canvas
      ref={canvasRef}
      width={WIDTH}
      height={HEIGHT}
      style={{ width: "100%", maxWidth: WIDTH, height: "auto", cursor: "pointer" }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    />
  );
}
