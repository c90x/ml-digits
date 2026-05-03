<script lang="ts">
	import type { DigitPrediction } from '$lib/digit-prediction';

	type Props = {
		prediction: DigitPrediction;
		showPrediction?: boolean;
	};

	let { prediction, showPrediction = false }: Props = $props();

	let confidence = $derived(
		prediction.predictedDigit >= 0 &&
			prediction.probabilities[prediction.predictedDigit] !== undefined
			? prediction.probabilities[prediction.predictedDigit]
			: 0
	);

	function isPredictedDigit(digit: number) {
		return showPrediction && digit === prediction.predictedDigit;
	}
</script>

<section class="w-full max-w-90">
	<div class="mb-5 flex min-h-5 items-center justify-between gap-4">
		<h3 class="text-sm font-semibold text-[#ccc]">
			{prediction.name}
		</h3>

		{#if prediction.isLoading}
			<span class="text-[0.8rem] text-orange-500">Loading</span>
		{:else if prediction.errorMessage}
			<span class="text-[0.8rem] text-red-400">Error</span>
		{:else if showPrediction && prediction.predictedDigit >= 0}
			<span class="text-[0.8rem] text-[#999]">
				<strong class="text-orange-500">{prediction.predictedDigit}</strong>
				({confidence.toFixed(2)})
			</span>
		{/if}
	</div>

	{#if prediction.errorMessage}
		<div
			class="flex h-45 items-center justify-center rounded-xl border border-[#333] px-4 text-sm text-red-400"
		>
			{prediction.errorMessage}
		</div>
	{:else}
		<div class="flex h-37.5 w-full items-end justify-center gap-1.5 sm:gap-2">
			{#each prediction.probabilities as probability, digit}
				<div class="flex w-8 flex-col items-center">
					<div class="flex h-30 w-full items-end justify-center">
						<div
							class={`relative min-h-0.5 w-full rounded-t transition-[height,background-color] duration-200 ease-out ${
								isPredictedDigit(digit) ? 'bg-orange-500' : 'bg-[#333]'
							}`}
							style="height: {probability * 100}%"
						>
							{#if probability > 0.01 && showPrediction}
								<span
									class="absolute -top-5 left-1/2 -translate-x-1/2 text-[0.7rem] font-semibold whitespace-nowrap text-orange-500"
								>
									{probability.toFixed(2)}
								</span>
							{/if}
						</div>
					</div>

					<span class="mt-1.5 text-[0.85rem] font-medium text-[#999]">
						{digit}
					</span>
				</div>
			{/each}
		</div>
	{/if}
</section>
