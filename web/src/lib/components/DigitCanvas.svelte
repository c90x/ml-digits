<script lang="ts">
	import { readCanvasDigit } from '$lib/digit-classifier';
	import DigitInputPreview from './DigitInputPreview.svelte';

	type Props = {
		size?: number;
		strokeWidth?: number;
		disabled?: boolean;
		showModelInputPreview?: boolean;
		onDrawingComplete?: (canvas: HTMLCanvasElement) => void;
	};

	let {
		size = 280,
		strokeWidth = 20,
		disabled = false,
		showModelInputPreview = false,
		onDrawingComplete
	}: Props = $props();

	let canvas: HTMLCanvasElement;
	let drawingContext: CanvasRenderingContext2D | undefined;
	let previewPixels = $state<Float32Array>();
	let isDrawing = false;

	export function clear() {
		fillCanvas();
	}

	$effect(() => {
		if (!canvas || drawingContext) return;

		drawingContext = canvas.getContext('2d') ?? undefined;

		if (!drawingContext) return;

		fillCanvas();

		drawingContext.lineCap = 'round';
		drawingContext.lineJoin = 'round';
		drawingContext.strokeStyle = 'black';
		drawingContext.lineWidth = strokeWidth;
	});

	$effect(() => {
		if (!canvas || !showModelInputPreview) return;

		previewPixels = readCanvasDigit(canvas);
	});

	function fillCanvas() {
		if (!drawingContext) return;

		drawingContext.fillStyle = 'white';
		drawingContext.fillRect(0, 0, size, size);
	}

	function getPointerPosition(event: PointerEvent) {
		const rect = canvas.getBoundingClientRect();

		return {
			x: ((event.clientX - rect.left) / rect.width) * size,
			y: ((event.clientY - rect.top) / rect.height) * size
		};
	}

	function startDrawing(event: PointerEvent) {
		if (disabled || !drawingContext) return;

		event.preventDefault();
		canvas.setPointerCapture(event.pointerId);

		const position = getPointerPosition(event);

		isDrawing = true;
		drawingContext.beginPath();
		drawingContext.moveTo(position.x, position.y);
	}

	function draw(event: PointerEvent) {
		if (!isDrawing || disabled || !drawingContext) return;

		event.preventDefault();

		const position = getPointerPosition(event);

		drawingContext.lineTo(position.x, position.y);
		drawingContext.stroke();
		drawingContext.beginPath();
		drawingContext.moveTo(position.x, position.y);
	}

	function stopDrawing(event: PointerEvent) {
		if (!isDrawing || !drawingContext) return;

		event.preventDefault();

		if (canvas.hasPointerCapture(event.pointerId)) {
			canvas.releasePointerCapture(event.pointerId);
		}

		isDrawing = false;
		drawingContext.beginPath();

		onDrawingComplete?.(canvas);
	}
</script>

<div
	class="relative cursor-crosshair touch-none overflow-hidden rounded-xl border-2 border-[#333] bg-white"
>
	<canvas
		bind:this={canvas}
		width={size}
		height={size}
		class="block aspect-square w-[min(70vw,260px)] rounded-[10px] bg-white sm:w-65"
		class:opacity-0={showModelInputPreview}
		onpointerdown={startDrawing}
		onpointermove={draw}
		onpointerup={stopDrawing}
		onpointerleave={stopDrawing}
		onpointercancel={stopDrawing}
	></canvas>

	{#if showModelInputPreview}
		<DigitInputPreview pixels={previewPixels} />
	{/if}
</div>
