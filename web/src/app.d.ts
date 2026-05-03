declare module '*.wasm?url' {
	const url: string;
	export default url;
}

declare global {
	namespace App {}
}

export {};
