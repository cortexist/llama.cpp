/**
 * Client-side video → frames extraction.
 *
 * The browser already ships a video decoder (the <video> element), so we sample
 * evenly-spaced frames into a <canvas> and export them as JPEG data URLs. The
 * caller attaches these as ordinary images, which ride the existing Vision
 * (image_url) path — no server-side video decoder required.
 */

const DEFAULT_MAX_FRAMES = 32; // ~1 fps up to ~30s (aligns with the 30s audio cap); bounds tokens
const MAX_DIMENSION = 512; // downscale longest side to keep token count modest

function seekTo(video: HTMLVideoElement, time: number): Promise<void> {
	return new Promise((resolve) => {
		const onSeeked = () => {
			video.removeEventListener('seeked', onSeeked);
			resolve();
		};
		video.addEventListener('seeked', onSeeked);
		video.currentTime = time;
	});
}

export interface VideoFramesResult {
	/** Extracted frames as JPEG data URLs (`data:image/jpeg;base64,...`). */
	frames: string[];
	/** Source video duration in seconds (for correct frame-player timing). */
	durationSec: number;
}

/**
 * Decode `file` in-browser and return up to `maxFrames` evenly-spaced frames as
 * JPEG data URLs. Frames are roughly 1 fps, capped at `maxFrames`. The video
 * duration is returned so a frame player can reproduce the original timing.
 */
export async function convertVideoToFrames(
	file: File,
	maxFrames: number = DEFAULT_MAX_FRAMES
): Promise<VideoFramesResult> {
	const url = URL.createObjectURL(file);
	const video = document.createElement('video');
	video.muted = true;
	video.preload = 'auto';
	video.src = url;

	try {
		await new Promise<void>((resolve, reject) => {
			video.onloadeddata = () => resolve();
			video.onerror = () => reject(new Error('Failed to load video'));
		});

		const duration = Number.isFinite(video.duration) ? video.duration : 0;
		const vw = video.videoWidth;
		const vh = video.videoHeight;
		if (vw === 0 || vh === 0) {
			throw new Error('Video has no visual track');
		}

		// ~1 fps, capped at maxFrames; at least 1 frame for very short/zero-duration clips
		const frameCount = Math.max(1, Math.min(maxFrames, Math.ceil(duration) || 1));

		const scale = Math.min(1, MAX_DIMENSION / Math.max(vw, vh));
		const canvas = document.createElement('canvas');
		canvas.width = Math.max(1, Math.round(vw * scale));
		canvas.height = Math.max(1, Math.round(vh * scale));
		const ctx = canvas.getContext('2d');
		if (!ctx) {
			throw new Error('Failed to get 2D canvas context');
		}

		const frames: string[] = [];
		for (let i = 0; i < frameCount; i++) {
			// sample at the midpoint of each evenly-spaced segment (avoids the often-black first/last frame)
			const t = duration > 0 ? (duration * (i + 0.5)) / frameCount : 0;
			await seekTo(video, t);
			ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
			frames.push(canvas.toDataURL('image/jpeg', 0.85));
		}

		return { frames, durationSec: duration };
	} finally {
		URL.revokeObjectURL(url);
		video.removeAttribute('src');
		video.load();
	}
}
