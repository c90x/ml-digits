<script lang="ts">
	import type { DigitPrediction } from '$lib/digit-prediction';
	import ModelResultChart from './ModelResultChart.svelte';

	type Props = {
		predictions?: DigitPrediction[];
		showPredictions?: boolean;
	};

	let { predictions = [], showPredictions = false }: Props = $props();
</script>

<section class="flex w-full flex-col items-center">
	<h2 class="mb-4 text-base font-semibold text-[#ccc]">Probability results</h2>

	<div class="flex w-full flex-col items-center gap-8">
		{#each predictions as prediction (prediction.id)}
			<ModelResultChart {prediction} showPrediction={showPredictions} />
		{/each}
	</div>
</section>
