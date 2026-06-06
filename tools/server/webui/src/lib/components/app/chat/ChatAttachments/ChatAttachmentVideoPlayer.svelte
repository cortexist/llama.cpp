<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX } from '@lucide/svelte';

	interface Props {
		// JPEG data URLs of the frames actually sent to the model
		frames: string[];
		// Source video duration (s); used so playback timing matches the original
		durationSec?: number;
		// Video audio track as a base64 16 kHz mono WAV (played in sync with the frames)
		audioWavBase64?: string;
		class?: string;
		// Tailwind sizing for the frame image (compact inline vs. large in the modal)
		imgClass?: string;
	}

	let {
		frames,
		durationSec = 0,
		audioWavBase64,
		class: className = '',
		imgClass = 'max-h-64 w-auto max-w-full'
	}: Props = $props();

	let audioUrl = $derived(audioWavBase64 ? `data:audio/wav;base64,${audioWavBase64}` : undefined);

	// currentTimeSec is the single clock; the displayed frame follows it. The clock
	// is advanced by the <audio> element when present, otherwise by a rAF timer so
	// frame-only videos still play at the right pace (stuttering on purpose).
	let currentTimeSec = $state(0);
	let playing = $state(false);
	let audioEl: HTMLAudioElement | undefined = $state();
	let muted = $state(false);
	let raf = 0;
	let lastTs = 0;

	function toggleMute() {
		muted = !muted;
		if (audioEl) audioEl.muted = muted;
	}

	let index = $derived(
		frames.length > 0 && durationSec > 0
			? Math.min(frames.length - 1, Math.floor((currentTimeSec / durationSec) * frames.length))
			: 0
	);

	function fmtTime(totalSeconds: number): string {
		const s = Math.max(0, Math.floor(totalSeconds));
		const m = Math.floor(s / 60);
		return `${m}:${String(s % 60).padStart(2, '0')}`;
	}

	function cancelRaf() {
		if (raf) cancelAnimationFrame(raf);
		raf = 0;
		lastTs = 0;
	}

	function stop() {
		playing = false;
		cancelRaf();
		audioEl?.pause();
	}

	function tick(ts: number) {
		if (!playing) return;

		if (audioEl && audioUrl) {
			currentTimeSec = audioEl.currentTime;
			if (audioEl.ended || currentTimeSec >= durationSec) {
				stop();
				return;
			}
		} else {
			if (!lastTs) lastTs = ts;
			currentTimeSec += (ts - lastTs) / 1000;
			lastTs = ts;
			if (currentTimeSec >= durationSec) {
				currentTimeSec = durationSec;
				stop();
				return;
			}
		}

		raf = requestAnimationFrame(tick);
	}

	function play() {
		if (frames.length <= 1 || durationSec <= 0) return;
		if (currentTimeSec >= durationSec) currentTimeSec = 0; // restart if at the end
		playing = true;
		lastTs = 0;
		if (audioEl && audioUrl) {
			audioEl.currentTime = currentTimeSec;
			void audioEl.play().catch(() => {});
		}
		raf = requestAnimationFrame(tick);
	}

	function toggle() {
		if (playing) stop();
		else play();
	}

	function seekTo(t: number) {
		stop();
		currentTimeSec = Math.max(0, Math.min(durationSec, t));
		if (audioEl) audioEl.currentTime = currentTimeSec;
	}

	$effect(() => () => {
		cancelRaf();
		audioEl?.pause();
	});
</script>

<div class="flex flex-col items-center gap-2 {className}">
	{#if frames.length > 0}
		<img
			src={frames[index]}
			alt={`Video frame ${index + 1} of ${frames.length}`}
			class="rounded-md object-contain {imgClass}"
		/>

		{#if audioUrl}
			<audio bind:this={audioEl} src={audioUrl} {muted} preload="auto"></audio>
		{/if}

		<div class="flex w-full items-center justify-center gap-2">
			<Button
				variant="ghost"
				size="icon"
				onclick={() => seekTo(0)}
				disabled={currentTimeSec <= 0}
				title="Jump to start"
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
				onclick={() => seekTo(durationSec)}
				disabled={index >= frames.length - 1}
				title="Jump to end"
			>
				<SkipForward class="h-4 w-4" />
			</Button>

			<span class="text-muted-foreground ml-2 text-xs tabular-nums">
				{fmtTime(currentTimeSec)} / {fmtTime(durationSec)}
			</span>

			{#if audioUrl}
				<Button
					variant="ghost"
					size="icon"
					onclick={toggleMute}
					title={muted ? 'Unmute audio' : 'Mute audio'}
				>
					{#if muted}
						<VolumeX class="h-4 w-4" />
					{:else}
						<Volume2 class="h-4 w-4" />
					{/if}
				</Button>
			{/if}
		</div>
	{/if}
</div>
