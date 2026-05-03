<script lang="ts">
	import DigitCanvas from '$lib/components/DigitCanvas.svelte';
	import ModelSettingsDialog from '$lib/components/ModelSettingsDialog.svelte';
	import ModelResultsPanel from '$lib/components/ModelResultsPanel.svelte';
	import { DigitClassifier, type DigitModel } from '$lib/digit-classifier';
	import {
		createDigitPrediction,
		markDigitPredictionComplete,
		markDigitPredictionFailed,
		markDigitPredictionLoading,
		markDigitPredictionReady,
		resetDigitPrediction,
		type DigitPrediction
	} from '$lib/digit-prediction';
	import { onMount } from 'svelte';
	import digitModelsJson from './models.json';

	const DRAWING_CANVAS_SIZE = 280;
	const DRAWING_STROKE_WIDTH = 20;
	const DEFAULT_MODEL_ID = '10_final_model_final';

	const digitModels: DigitModel[] = digitModelsJson;
	const modelGroups = [
		{ title: 'Final Model', models: getModelsById(['10_final_model_final']) },
		{
			title: '1. (Logistic Regression) Learning Rates (Epoch 50)',
			models: getModelsByPrefix('02_logistic_regression_lr')
		},
		{
			title: '2. (Logistic Regression) Optimizers (Epoch 50)',
			models: getModelsByPrefix('04_logistic_regression_optimizers')
		},
		{
			title: '3. (Neural Network) ReLU Layers (Epoch 50)',
			models: getModelsByPrefix('03_layers_relu')
		},
		{
			title: '4. (Convolutional Network) Convolution Layers (early stopping, max Epoch 50)',
			models: getModelsByPrefix('05_convolution_configs_adamw')
		}
	].filter((group) => group.models.length > 0);

	const digitClassifier = new DigitClassifier(digitModels);

	let digitCanvas: DigitCanvas;
	let currentCanvas: HTMLCanvasElement | null = null;
	let enabledModelIds = $state([DEFAULT_MODEL_ID]);
	let hasDrawing = $state(false);
	let showModelSettings = $state(false);
	let showModelInputPreview = $state(false);
	let predictionRunId = 0;
	let predictions = $state<DigitPrediction[]>(
		getEnabledModels().map((model) => createDigitPrediction(model, true))
	);

	onMount(() => {
		void preloadDigitModels();
	});

	async function preloadDigitModels() {
		await Promise.all(getEnabledModels().map(preloadDigitModel));
	}

	async function preloadDigitModel(model: DigitModel) {
		try {
			await digitClassifier.load(model.id);

			if (!hasDrawing) {
				updatePrediction(model.id, markDigitPredictionReady);
			}
		} catch (error) {
			if (!hasDrawing) {
				updatePrediction(model.id, (prediction) => markDigitPredictionFailed(prediction, error));
			}
		}
	}

	function updatePrediction(
		modelId: string,
		update: (prediction: DigitPrediction) => DigitPrediction
	) {
		predictions = predictions.map((prediction) =>
			prediction.id === modelId ? update(prediction) : prediction
		);
	}

	function getModelsById(modelIds: string[]) {
		return digitModels.filter((model) => modelIds.includes(model.id));
	}

	function getModelsByPrefix(modelIdPrefix: string) {
		return digitModels.filter((model) => model.id.startsWith(modelIdPrefix));
	}

	function getEnabledModels() {
		return digitModels.filter((model) => enabledModelIds.includes(model.id));
	}

	function isModelEnabled(modelId: string) {
		return enabledModelIds.includes(modelId);
	}

	function syncEnabledPredictions(loadingModelId?: string) {
		const predictionById = new Map(predictions.map((prediction) => [prediction.id, prediction]));

		predictions = getEnabledModels().map(
			(model) =>
				predictionById.get(model.id) ?? createDigitPrediction(model, model.id === loadingModelId)
		);
	}

	function setModelEnabled(model: DigitModel, enabled: boolean) {
		if (enabled === isModelEnabled(model.id)) return;

		enabledModelIds = enabled
			? [...enabledModelIds, model.id]
			: enabledModelIds.filter((modelId) => modelId !== model.id);

		syncEnabledPredictions(enabled ? model.id : undefined);

		if (!enabled) return;

		if (hasDrawing && currentCanvas) {
			void predictDigitModel(model, currentCanvas, predictionRunId);
		} else {
			void preloadDigitModel(model);
		}
	}

	function disableAllModels() {
		enabledModelIds = [];
		predictions = [];
		predictionRunId += 1;
	}

	function clearCanvas() {
		showModelInputPreview = false;
		digitCanvas.clear();
		currentCanvas = null;
		hasDrawing = false;
		predictionRunId += 1;
		predictions = predictions.map(resetDigitPrediction);
	}

	function startModelInputPreview(event: PointerEvent) {
		event.preventDefault();
		(event.currentTarget as HTMLButtonElement | null)?.setPointerCapture(event.pointerId);
		showModelInputPreview = true;
	}

	function stopModelInputPreview(event: PointerEvent) {
		event.preventDefault();
		showModelInputPreview = false;
	}

	async function handleDrawingComplete(canvas: HTMLCanvasElement) {
		hasDrawing = true;
		currentCanvas = canvas;

		const runId = predictionRunId + 1;
		const loadingPredictions = getEnabledModels().map((model) => {
			const prediction = predictions.find((candidate) => candidate.id === model.id);

			return markDigitPredictionLoading(prediction ?? createDigitPrediction(model));
		});

		predictionRunId = runId;
		predictions = loadingPredictions;

		const completedPredictions = await Promise.all(
			loadingPredictions.map(async (prediction) => {
				try {
					const probabilities = await digitClassifier.predict(prediction.id, canvas);

					return markDigitPredictionComplete(prediction, probabilities);
				} catch (error) {
					return markDigitPredictionFailed(prediction, error);
				}
			})
		);

		if (runId === predictionRunId) {
			const currentPredictionById = new Map(
				predictions.map((prediction) => [prediction.id, prediction])
			);
			const completedPredictionById = new Map(
				completedPredictions.map((prediction) => [prediction.id, prediction])
			);

			predictions = getEnabledModels().map(
				(model) =>
					completedPredictionById.get(model.id) ??
					currentPredictionById.get(model.id) ??
					createDigitPrediction(model)
			);
		}
	}

	async function predictDigitModel(model: DigitModel, canvas: HTMLCanvasElement, runId: number) {
		try {
			const probabilities = await digitClassifier.predict(model.id, canvas);

			if (runId === predictionRunId && isModelEnabled(model.id)) {
				updatePrediction(model.id, (prediction) =>
					markDigitPredictionComplete(prediction, probabilities)
				);
			}
		} catch (error) {
			if (runId === predictionRunId && isModelEnabled(model.id)) {
				updatePrediction(model.id, (prediction) => markDigitPredictionFailed(prediction, error));
			}
		}
	}
</script>

<svelte:head>
	<title>ML Digits Classifier</title>
</svelte:head>

<main class="min-h-screen bg-[#0a0a0a] px-4 py-8 text-center text-[#e0e0e0] sm:px-6 sm:py-10">
	<section class="mx-auto w-full max-w-225">
		<header class="mb-5 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
			<h1 class="col-start-2 text-[2rem] leading-tight font-bold text-white sm:text-[2.2rem]">
				ml-digits
			</h1>

			<button
				type="button"
				class="col-start-3 justify-self-end rounded-lg border border-[#444] bg-[#202020] px-3 py-2 text-sm font-semibold text-[#eee] transition hover:bg-[#2a2a2a] active:scale-[0.98]"
				aria-haspopup="dialog"
				aria-expanded={showModelSettings}
				onclick={() => (showModelSettings = true)}
			>
				Settings
			</button>
		</header>

		<div
			class="mx-auto grid w-full max-w-225 grid-cols-1 gap-9 rounded-2xl border border-[#2a2a2a] bg-[#141414] p-6 sm:p-8 md:grid-cols-[260px_minmax(320px,360px)] md:items-start md:justify-center md:gap-12 lg:p-10"
		>
			<section class="flex w-full flex-col items-center">
				<h2 class="mb-4 text-base font-semibold text-[#ccc]">Input</h2>

				<DigitCanvas
					bind:this={digitCanvas}
					size={DRAWING_CANVAS_SIZE}
					strokeWidth={DRAWING_STROKE_WIDTH}
					{showModelInputPreview}
					onDrawingComplete={handleDrawingComplete}
				/>

				<div class="mt-4 flex items-center gap-3">
					<button
						type="button"
						class="rounded-lg bg-orange-500 px-10 py-2.5 text-[0.95rem] font-semibold text-white transition hover:bg-orange-600 active:scale-[0.98]"
						onclick={clearCanvas}
					>
						Clear
					</button>

					<button
						type="button"
						aria-label="Hold to preview model input"
						title="Hold to preview model input"
						class="grid size-11 touch-none place-items-center rounded-lg border border-[#444] bg-[#202020] text-xs font-bold tracking-tight text-[#eee] transition select-none hover:bg-[#2a2a2a] active:scale-[0.98]"
						onpointerdown={startModelInputPreview}
						onpointerup={stopModelInputPreview}
						onpointercancel={stopModelInputPreview}
						onlostpointercapture={stopModelInputPreview}
						oncontextmenu={(event) => event.preventDefault()}
					>
						28
					</button>
				</div>
			</section>

			<ModelResultsPanel {predictions} showPredictions={hasDrawing} />
		</div>

		<footer class="mt-8 text-[0.85rem] text-[#555]">
			Browser based demo using ONNX Runtime for web.
		</footer>
	</section>

	{#if showModelSettings}
		<ModelSettingsDialog
			groups={modelGroups}
			{enabledModelIds}
			onToggle={setModelEnabled}
			onDisableAll={disableAllModels}
			onClose={() => (showModelSettings = false)}
		/>
	{/if}
</main>

<style>
	:global(body) {
		margin: 0;
	}
</style>
