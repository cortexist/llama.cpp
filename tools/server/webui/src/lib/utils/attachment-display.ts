import { AttachmentType, FileTypeCategory, SpecialFileType } from '$lib/enums';
import {
	getFileTypeCategory,
	getFileTypeCategoryByExtension,
	isImageFile,
	isVideoFile
} from '$lib/utils';
import type {
	AttachmentDisplayItemsOptions,
	ChatUploadedFile,
	DatabaseMessageExtra
} from '$lib/types';

/**
 * Check if an uploaded file is an MCP prompt
 */
function isMcpPromptUpload(file: ChatUploadedFile): boolean {
	return file.type === SpecialFileType.MCP_PROMPT && !!file.mcpPrompt;
}

/**
 * Check if an attachment is an MCP prompt
 */
function isMcpPromptAttachment(attachment: DatabaseMessageExtra): boolean {
	return attachment.type === AttachmentType.MCP_PROMPT;
}

/**
 * Check if an attachment is an MCP resource
 */
function isMcpResourceAttachment(attachment: DatabaseMessageExtra): boolean {
	return attachment.type === AttachmentType.MCP_RESOURCE;
}

/**
 * Gets the file type category from an uploaded file, checking both MIME type and extension
 */
function getUploadedFileCategory(file: ChatUploadedFile): FileTypeCategory | null {
	const categoryByMime = getFileTypeCategory(file.type);

	if (categoryByMime) {
		return categoryByMime;
	}

	return getFileTypeCategoryByExtension(file.name);
}

/**
 * Creates a unified list of display items from uploaded files and stored attachments.
 * Items are returned in reverse order (newest first).
 */
export function getAttachmentDisplayItems(
	options: AttachmentDisplayItemsOptions
): ChatAttachmentDisplayItem[] {
	const { uploadedFiles = [], attachments = [] } = options;
	const items: ChatAttachmentDisplayItem[] = [];

	// Add uploaded files (ChatForm)
	for (const file of uploadedFiles) {
		items.push({
			id: file.id,
			name: file.name,
			size: file.size,
			preview: file.preview,
			isImage: getUploadedFileCategory(file) === FileTypeCategory.IMAGE,
			isVideo: getUploadedFileCategory(file) === FileTypeCategory.VIDEO,
			isMcpPrompt: isMcpPromptUpload(file),
			isLoading: file.isLoading,
			loadError: file.loadError,
			uploadedFile: file,
			textContent: file.textContent,
			videoFrames: file.videoFrames,
			videoDurationSec: file.videoDurationSec,
			videoAudioWavBase64: file.videoAudioWavBase64
		});
	}

	// Add stored attachments (ChatMessage)
	for (const [index, attachment] of attachments.entries()) {
		const isImage = isImageFile(attachment);
		const isVideo = isVideoFile(attachment);
		const isMcpPrompt = isMcpPromptAttachment(attachment);
		const isMcpResource = isMcpResourceAttachment(attachment);
		const videoFrames = isVideo && 'frames' in attachment ? attachment.frames : undefined;

		items.push({
			id: `attachment-${index}`,
			name: attachment.name,
			preview: isImage && 'base64Url' in attachment ? attachment.base64Url : videoFrames?.[0],
			isImage,
			isVideo,
			isMcpPrompt,
			isMcpResource,
			attachment,
			attachmentIndex: index,
			textContent: 'content' in attachment ? attachment.content : undefined,
			videoFrames,
			videoDurationSec: isVideo && 'durationSec' in attachment ? attachment.durationSec : undefined,
			videoAudioWavBase64:
				isVideo && 'audioWavBase64' in attachment ? attachment.audioWavBase64 : undefined
		});
	}

	return items.reverse();
}
