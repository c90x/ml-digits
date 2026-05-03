import * as ort from 'onnxruntime-web/wasm';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';

export const DIGIT_IMAGE_SIZE = 28;
const DIGIT_OUTPUT_COUNT = 10;

export type DigitModel = {
	id: string;
	name: string;
	url: string;
};

type LoadedDigitModel = {
	session: ort.InferenceSession;
	inputName: string;
	outputName: string;
};

let onnxRuntimeConfigured = false;

export class DigitClassifier {
	private readonly modelById: Map<string, DigitModel>;
	private readonly loadedModelById = new Map<string, LoadedDigitModel>();
	private readonly modelLoadById = new Map<string, Promise<LoadedDigitModel>>();

	constructor(models: DigitModel[]) {
		configureOnnxRuntime();

		this.modelById = new Map(models.map((model) => [model.id, model]));
	}

	async load(modelId: string) {
		await this.getLoadedModel(modelId);
	}

	async predict(modelId: string, sourceCanvas: HTMLCanvasElement) {
		const model = await this.getLoadedModel(modelId);
		const inputTensor = new ort.Tensor('float32', readCanvasDigit(sourceCanvas), [
			1,
			1,
			DIGIT_IMAGE_SIZE,
			DIGIT_IMAGE_SIZE
		]);
		const outputByName = await model.session.run({ [model.inputName]: inputTensor });
		const outputTensor = outputByName[model.outputName] ?? Object.values(outputByName)[0];

		if (!outputTensor) {
			throw new Error('ONNX model returned no outputs');
		}

		return softmax(readDigitScores(outputTensor));
	}

	private async getLoadedModel(modelId: string) {
		const loadedModel = this.loadedModelById.get(modelId);

		if (loadedModel) return loadedModel;

		const existingLoad = this.modelLoadById.get(modelId);

		if (existingLoad) return existingLoad;

		const model = this.modelById.get(modelId);

		if (!model) {
			throw new Error(`Unknown model: ${modelId}`);
		}

		const modelLoad = this.loadModel(model).finally(() => {
			this.modelLoadById.delete(modelId);
		});

		this.modelLoadById.set(modelId, modelLoad);

		return modelLoad;
	}

	private async loadModel(model: DigitModel): Promise<LoadedDigitModel> {
		const response = await fetch(model.url);

		if (!response.ok) {
			throw new Error(
				`Failed to fetch ${model.name} from ${model.url}: ${response.status} ${response.statusText}`
			);
		}

		const session = await ort.InferenceSession.create(await response.arrayBuffer(), {
			executionProviders: ['wasm']
		});
		const inputName = session.inputNames[0];
		const outputName = session.outputNames[0];

		if (!inputName || !outputName) {
			throw new Error(`${model.name} does not expose an input and output tensor`);
		}

		const loadedModel = { session, inputName, outputName };

		this.loadedModelById.set(model.id, loadedModel);

		return loadedModel;
	}
}

function configureOnnxRuntime() {
	if (onnxRuntimeConfigured) return;

	ort.env.wasm.numThreads = 1;
	ort.env.wasm.wasmPaths = { wasm: ortWasmUrl };
	onnxRuntimeConfigured = true;
}

export function readCanvasDigit(sourceCanvas: HTMLCanvasElement) {
	const digitCanvas = document.createElement('canvas');
	digitCanvas.width = DIGIT_IMAGE_SIZE;
	digitCanvas.height = DIGIT_IMAGE_SIZE;

	const digitContext = digitCanvas.getContext('2d');

	if (!digitContext) {
		throw new Error('Could not read canvas pixels');
	}

	digitContext.drawImage(sourceCanvas, 0, 0, DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE);

	const imageData = digitContext.getImageData(0, 0, DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE);
	const digitPixels = new Float32Array(DIGIT_IMAGE_SIZE * DIGIT_IMAGE_SIZE);

	for (let pixel = 0; pixel < digitPixels.length; pixel += 1) {
		const offset = pixel * 4;
		const red = imageData.data[offset];
		const green = imageData.data[offset + 1];
		const blue = imageData.data[offset + 2];
		const alpha = imageData.data[offset + 3] / 255;
		const brightness = (red + green + blue) / (3 * 255);

		digitPixels[pixel] = alpha * (1 - brightness);
	}

	return digitPixels;
}

function readDigitScores(outputTensor: ort.Tensor) {
	const scores = Array.from(outputTensor.data as ArrayLike<number>, Number);

	if (scores.length < DIGIT_OUTPUT_COUNT) {
		throw new Error(`Expected at least 10 output values, received ${scores.length}`);
	}

	return scores.slice(0, DIGIT_OUTPUT_COUNT);
}

function softmax(scores: number[]) {
	const maxScore = Math.max(...scores);
	const exponentials = scores.map((score) => Math.exp(score - maxScore));
	const exponentialTotal = exponentials.reduce((total, value) => total + value, 0);

	return exponentials.map((value) => value / exponentialTotal);
}
