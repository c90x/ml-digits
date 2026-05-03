<script lang="ts">
	import type { DigitModel } from '$lib/digit-classifier';

	type ModelGroup = {
		title: string;
		models: DigitModel[];
	};

	type Props = {
		groups: ModelGroup[];
		enabledModelIds: string[];
		onToggle: (model: DigitModel, enabled: boolean) => void;
		onDisableAll: () => void;
		onClose: () => void;
	};

	let { groups, enabledModelIds, onToggle, onDisableAll, onClose }: Props = $props();

	function isEnabled(modelId: string) {
		return enabledModelIds.includes(modelId);
	}

	function handleBackdropClick(event: MouseEvent) {
		if (event.target === event.currentTarget) onClose();
	}

	function handleToggle(model: DigitModel, event: Event) {
		onToggle(model, (event.currentTarget as HTMLInputElement).checked);
	}
</script>

<svelte:window onkeydown={(event) => event.key === 'Escape' && onClose()} />

<div
	class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 text-left sm:p-6"
	onclick={handleBackdropClick}
	role="presentation"
>
	<div
		class="flex max-h-[calc(100dvh-1.5rem)] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-[#333] bg-[#141414] shadow-2xl sm:max-h-[min(86vh,760px)]"
		role="dialog"
		aria-modal="true"
		aria-labelledby="model-settings-title"
	>
		<header
			class="flex items-center justify-between gap-4 border-b border-[#2a2a2a] px-5 py-4 sm:px-6"
		>
			<div>
				<p class="text-xs font-semibold tracking-[0.2em] text-orange-500 uppercase">Settings</p>
				<h2 id="model-settings-title" class="mt-1 text-xl font-bold text-white">Models</h2>
			</div>

			<div class="flex shrink-0 items-center gap-2">
				<button
					type="button"
					class="rounded-lg border border-[#5a3028] bg-[#2a1714] px-3 py-2 text-sm font-semibold text-orange-200 transition hover:bg-[#3a201b] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50"
					disabled={enabledModelIds.length === 0}
					onclick={onDisableAll}
				>
					Disable all
				</button>

				<button
					type="button"
					class="rounded-lg border border-[#444] bg-[#202020] px-3 py-2 text-sm font-semibold text-[#eee] transition hover:bg-[#2a2a2a] active:scale-[0.98]"
					onclick={onClose}
				>
					Close
				</button>
			</div>
		</header>

		<div class="min-h-0 overflow-y-auto px-5 py-5 sm:px-6">
			<div class="grid gap-5">
				{#each groups as group (group.title)}
					<section class="rounded-xl border border-[#2a2a2a] bg-[#181818] p-4">
						<h3 class="mb-3 text-sm font-semibold text-[#ccc]">{group.title}</h3>

						<div class="grid gap-2">
							{#each group.models as model (model.id)}
								<label
									class="flex cursor-pointer items-center justify-between gap-4 rounded-lg border border-[#2a2a2a] bg-[#202020] px-3 py-3 transition hover:border-[#444]"
								>
									<span class="text-sm font-medium text-[#eee]">{model.name}</span>

									<input
										type="checkbox"
										class="peer sr-only"
										checked={isEnabled(model.id)}
										onchange={(event) => handleToggle(model, event)}
									/>

									<span
										class={`flex h-6 w-11 shrink-0 items-center rounded-full p-1 transition peer-focus-visible:ring-2 peer-focus-visible:ring-orange-500 ${
											isEnabled(model.id) ? 'justify-end bg-orange-500' : 'justify-start bg-[#333]'
										}`}
										aria-hidden="true"
									>
										<span class="size-4 rounded-full bg-white"></span>
									</span>
								</label>
							{/each}
						</div>
					</section>
				{/each}
			</div>
		</div>
	</div>
</div>
