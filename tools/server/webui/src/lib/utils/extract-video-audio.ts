/**
 * Client-side extraction of a video's audio track into a 16 kHz mono WAV.
 *
 * The browser decodes the container's audio with the Web Audio API; we resample
 * to 16 kHz mono (what Gemma 4's audio encoder expects) and encode a small WAV
 * that is both sent to the model (`input_audio`) and played back in the player.
 * Returns null when the file has no decodable audio track.
 */

const TARGET_SAMPLE_RATE = 16000;
const DEFAULT_MAX_SECONDS = 30; // Gemma 4 audio is capped at ~30s

function floatTo16BitPCM(samples: Float32Array): ArrayBuffer {
	const buffer = new ArrayBuffer(samples.length * 2);
	const view = new DataView(buffer);
	for (let i = 0; i < samples.length; i++) {
		const s = Math.max(-1, Math.min(1, samples[i]));
		view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
	}
	return buffer;
}

function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
	const pcm = floatTo16BitPCM(samples);
	const buffer = new ArrayBuffer(44 + pcm.byteLength);
	const view = new DataView(buffer);
	const writeStr = (offset: number, str: string) => {
		for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
	};

	writeStr(0, 'RIFF');
	view.setUint32(4, 36 + pcm.byteLength, true);
	writeStr(8, 'WAVE');
	writeStr(12, 'fmt ');
	view.setUint32(16, 16, true); // PCM chunk size
	view.setUint16(20, 1, true); // PCM format
	view.setUint16(22, 1, true); // mono
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, sampleRate * 2, true); // byte rate (mono, 16-bit)
	view.setUint16(32, 2, true); // block align
	view.setUint16(34, 16, true); // bits per sample
	writeStr(36, 'data');
	view.setUint32(40, pcm.byteLength, true);
	new Uint8Array(buffer, 44).set(new Uint8Array(pcm));
	return buffer;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
	const bytes = new Uint8Array(buffer);
	let binary = '';
	const chunk = 0x8000;
	for (let i = 0; i < bytes.length; i += chunk) {
		binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
	}
	return btoa(binary);
}

export interface VideoAudioResult {
	/** Base64-encoded 16 kHz mono WAV (no data-URL prefix). */
	wavBase64: string;
	/** Audio duration in seconds (after the max-length cap). */
	durationSec: number;
}

export async function extractVideoAudioWav(
	file: File,
	maxSeconds: number = DEFAULT_MAX_SECONDS
): Promise<VideoAudioResult | null> {
	const AudioCtx: typeof AudioContext =
		window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
	if (!AudioCtx) return null;

	const arrayBuffer = await file.arrayBuffer();
	const decodeCtx = new AudioCtx();
	let decoded: AudioBuffer;
	try {
		decoded = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
	} catch {
		// No audio track, or a codec the browser can't decode
		await decodeCtx.close();
		return null;
	}
	await decodeCtx.close();

	if (decoded.length === 0 || decoded.duration === 0) return null;

	const seconds = Math.min(decoded.duration, maxSeconds);
	const frameCount = Math.ceil(seconds * TARGET_SAMPLE_RATE);

	// Resample to 16 kHz mono via an offline render
	const offline = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE);
	const source = offline.createBufferSource();
	source.buffer = decoded;
	source.connect(offline.destination);
	source.start();
	const rendered = await offline.startRendering();

	const mono = rendered.getChannelData(0);
	const wav = encodeWav(mono, TARGET_SAMPLE_RATE);

	return { wavBase64: arrayBufferToBase64(wav), durationSec: seconds };
}
