<script lang="ts">
	import { DIGIT_IMAGE_SIZE } from '$lib/digit-classifier';

	type Props = {
		pixels?: Float32Array;
	};

	let { pixels }: Props = $props();
	let canvas: HTMLCanvasElement;

	$effect(() => {
		if (!canvas || !pixels) return;

		const context = canvas.getContext('2d');

		if (!context) return;

		const imageData = context.createImageData(DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE);

		for (let pixel = 0; pixel < pixels.length; pixel += 1) {
			const value = Math.round(pixels[pixel] * 255);
			const offset = pixel * 4;

			imageData.data[offset] = value;
			imageData.data[offset + 1] = value;
			imageData.data[offset + 2] = value;
			imageData.data[offset + 3] = 255;
		}

		context.putImageData(imageData, 0, 0);
	});
</script>

<canvas
	bind:this={canvas}
	width={DIGIT_IMAGE_SIZE}
	height={DIGIT_IMAGE_SIZE}
	class="pointer-events-none absolute inset-0 h-full w-full rounded-[10px] bg-black [image-rendering:pixelated]"
></canvas>
