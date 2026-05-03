import type { DigitModel } from './digit-classifier';

const DIGIT_COUNT = 10;

export type DigitPrediction = {
	id: string;
	name: string;
	probabilities: number[];
	predictedDigit: number;
	isLoading: boolean;
	errorMessage: string | null;
};

export function createDigitPrediction(model: DigitModel, isLoading = false): DigitPrediction {
	return {
		id: model.id,
		name: model.name,
		probabilities: emptyProbabilities(),
		predictedDigit: -1,
		isLoading,
		errorMessage: null
	};
}

export function resetDigitPrediction(prediction: DigitPrediction): DigitPrediction {
	return {
		...prediction,
		probabilities: emptyProbabilities(),
		predictedDigit: -1,
		isLoading: false,
		errorMessage: null
	};
}

export function markDigitPredictionReady(prediction: DigitPrediction): DigitPrediction {
	return {
		...prediction,
		isLoading: false,
		errorMessage: null
	};
}

export function markDigitPredictionLoading(prediction: DigitPrediction): DigitPrediction {
	return {
		...prediction,
		isLoading: true,
		errorMessage: null
	};
}

export function markDigitPredictionComplete(
	prediction: DigitPrediction,
	probabilities: number[]
): DigitPrediction {
	return {
		...prediction,
		probabilities,
		predictedDigit: getPredictedDigit(probabilities),
		isLoading: false,
		errorMessage: null
	};
}

export function markDigitPredictionFailed(
	prediction: DigitPrediction,
	error: unknown
): DigitPrediction {
	return {
		...prediction,
		isLoading: false,
		errorMessage: getErrorMessage(error)
	};
}

function emptyProbabilities() {
	return Array(DIGIT_COUNT).fill(0);
}

function getPredictedDigit(probabilities: number[]) {
	return probabilities.indexOf(Math.max(...probabilities));
}

function getErrorMessage(error: unknown) {
	return error instanceof Error ? error.message : 'Prediction failed';
}
