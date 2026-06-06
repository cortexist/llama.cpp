<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Play, Pause, SkipBack, SkipForward } from '@lucide/svelte';

	interface Props {
		// JPEG data URLs of the frames actually sent to the model
		frames: string[];
		// Source video duration (s); used so playback timing matches the original
		durationSec?: number;
		class?: string;
		// Tailwind sizing for the frame image (compact inline vs. large in the modal)
		imgClass?: string;
	}

	let {
		frames,
		durationSec = 0,
		class: className = '',
		imgClass = 'max-h-64 w-auto max-w-full'
	}: Props = $props();

	let index = $state(0);
	let playing = $state(false);
	let timer: ReturnType<typeof setTimeout> | null = null;

	// Per-frame hold time. Stuttering on purpose: each frame is held for the slice
	// of real time it represents, so the developer sees exactly what the model got.
	let frameMs = $derived(
		frames.length > 1 && durationSec > 0 ? (durationSec * 1000) / frames.length : 1000
	);

	function clearTimer() {
		if (timer) {
			clearTimeout(timer);
			timer = null;
		}
	}

	function stop() {
		playing = false;
		clearTimer();
	}

	function scheduleNext() {
		clearTimer();
		timer = setTimeout(() => {
			if (index >= frames.length - 1) {
				stop();
				return;
			}
			index += 1;
			scheduleNext();
		}, frameMs);
	}

	function play() {
		if (frames.length <= 1) return;
		if (index >= frames.length - 1) index = 0; // restart if at the end
		playing = true;
		scheduleNext();
	}

	function toggle() {
		if (playing) stop();
		else play();
	}

	function toStart() {
		stop();
		index = 0;
	}

	function toEnd() {
		stop();
		index = frames.length - 1;
	}

	$effect(() => () => clearTimer());
</script>

<div class="flex flex-col items-center gap-2 {className}">
	{#if frames.length > 0}
		<img
			src={frames[index]}
			alt={`Video frame ${index + 1} of ${frames.length}`}
			class="rounded-md object-contain {imgClass}"
		/>

		<div class="flex w-full items-center justify-center gap-2">
			<Button
				variant="ghost"
				size="icon"
				onclick={toStart}
				disabled={index === 0}
				title="Jump to first frame"
			>
				<SkipBack class="h-4 w-4" />
			</Button>

			<Button variant="ghost" size="icon" onclick={toggle} title={playing ? 'Stop' : 'Play'}>
				{#if playing}
					<Pause class="h-4 w-4" />
				{:else}
					<Play class="h-4 w-4" />
				{/if}
			</Button>

			<Button
				variant="ghost"
				size="icon"
				onclick={toEnd}
				disabled={index >= frames.length - 1}
				title="Jump to last frame"
			>
				<SkipForward class="h-4 w-4" />
			</Button>

			<span class="text-muted-foreground ml-2 text-xs tabular-nums">
				frame {index + 1} / {frames.length}
			</span>
		</div>
	{/if}
</div>
