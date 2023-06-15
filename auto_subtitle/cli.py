import os
import ffmpeg
import argparse
import logging
import tempfile
import cv2
from typing import Iterator, TextIO


def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def write_srt(transcript: Iterator[dict], file: TextIO):
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True)} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True)}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]


try:
    import whisper
except ImportError as e:
    logging.error("Error importing 'whisper'. Please make sure it is installed.", e)
    exit(1)


output_dir = "./output"  # changed to "./output"
logging.basicConfig(level=logging.INFO)


def detect_faces(frame, face_cascade):
    """Detect faces in the frame using provided cascade classifier."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )


def extract_audio_from_video(video_paths):
    """Extracts audio from video files."""
    temp_dir = tempfile.gettempdir()
    audio_paths = {}
    for path in video_paths:
        logging.info(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")
        ffmpeg.input(path).output(output_path, acodec="pcm_s16le", ac=1, ar="16k").run(
            quiet=True, overwrite_output=True
        )
        audio_paths[path] = output_path
    return audio_paths


def save_face(frame, face,basename):
    """Saves the face from the frame to a jpg file."""
    x, y, w, h = face
    face_image = frame[y : y + h, x : x + w]
    face_output_path = os.path.join(output_dir, f"{basename}_face.jpg")
    cv2.imwrite(face_output_path, face_image)


def extract_webcam_coords(frame, facex, facey,basename):
    # Calculate the region of interest
    roi_size = int(frame.shape[1] / 4)
    roi_x = facex - roi_size // 2
    roi_y = facey - roi_size // 2

    # Ensure the ROI coordinates are within the frame boundaries
    roi_x = max(0, roi_x)
    roi_y = max(0, roi_y)
    roi_x2 = min(frame.shape[1], roi_x + roi_size)
    roi_y2 = min(frame.shape[0], roi_y + roi_size)

    # Create the ROI
    roi = frame[roi_y:roi_y2, roi_x:roi_x2]

    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Perform thresholding to detect white rectangular boxes
    _, threshold = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangular contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box coordinates of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Convert the local coordinates to global coordinates
    x += roi_x
    y += roi_y

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the image with the bounding box
    cv2.imwrite('output.jpg', frame)

    # Return the global coordinates of the bounding box
    return x, y, w, h



def crop_and_add_overlay(input_path, output_path, face_cascade):
    
    # save on generated folder
    base_name = os.path.splitext(os.path.basename(input_path))[0] 
 
    """Crops the input video around a detected face and adds an overlay."""
    cap = cv2.VideoCapture(input_path)
    success, frame = cap.read()
    if not success:
        logging.error("Failed to read input video.")
        return False

    faces = detect_faces(frame, face_cascade)

    if len(faces) == 0:
        logging.warning("No faces found in the input video.")
        return False

    logging.info(f"Found {len(faces)} faces in the input video.")

    biggest_face = max(faces, key=lambda f: f[2] * f[3])
    save_face(frame, biggest_face,base_name)

    # Draw rectangle on the biggest face
    (x, y, w, h) = biggest_face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    base_name = os.path.splitext(os.path.basename(input_path))[0] 
    x, y, h, w = extract_webcam_coords(frame, x,y,base_name)
    print("Coordinates of the biggest face: ", x, y, w, h)
    
    save_face(frame, (x, y, w, h),base_name+"_webcam")


    # Crop the original video to obtain the webcam video
    webcam_video = ffmpeg.input(input_path).crop(
        x=x, y=y, width=w, height=h
    )

    # save on generated folder# save on generated folder
    webcam_output_path = os.path.join(output_dir, f"{base_name}_webcam_video.mp4")

    ffmpeg.output(webcam_video, webcam_output_path).run(overwrite_output=True)

    # Crop the input video to the middle
    cropped_video = ffmpeg.input(input_path).filter("crop", "ih*(9/16)", "ih")

    # Overlay the webcam video on top of the cropped video
    output_video = cropped_video.overlay(webcam_video, x=x, y=y)

    # Extract the audio from the input video
    audio = ffmpeg.input(input_path).audio

    # Output the final video with the overlayed webcam video
    ffmpeg.output(output_video, audio, output_path, crf=21).run(overwrite_output=True)

    return True


def get_audio(paths):
    temp_dir = tempfile.gettempdir()
    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(output_path, acodec="pcm_s16le", ac=1, ar="16k").run(
            quiet=True, overwrite_output=True
        )

        audio_paths[path] = output_path

    return audio_paths


def get_subtitles(audio_paths, input_dir, transcribe):
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = os.path.join(output_dir, f"{filename(path)}.srt")

        logging.info(f"srt_path: {srt_path}")

        # Check if SRT file already exists
        if os.path.exists(srt_path):
            logging.info(f"Subtitle file already exists: {srt_path}")
        else:
            logging.info(
                f"Generating subtitles for {filename(path)}... This might take a while."
            )

            result = transcribe(audio_path)

            # Check if result is not None before attempting to use it
            if result is not None and "segments" in result:
                try:
                    with open(srt_path, "w", encoding="utf-8") as srt:
                        write_srt(result["segments"], file=srt)
                    logging.info(f"Subtitle file saved: {srt_path}")
                except Exception as e:
                    logging.error(
                        f"Error saving subtitles to file: {srt_path}. Error: {str(e)}"
                    )
            else:
                logging.error(f"Error generating subtitles for {filename(path)}.")

        subtitles_path[path] = srt_path

    return subtitles_path


def process_videos(
    input_dir,
    model_name,
    output_srt,
    srt_only,
    language,
    task,
    face_cascade,
):
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        exit(1)

    # Create output directories
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating output directory: {str(e)}")
        exit(1)

    if model_name.endswith(".en"):
        language = "en"
    elif language != "auto":
        language = language

    videos = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.lower().endswith(".mp4")
    ]
    audios = get_audio(videos)
    subtitles = get_subtitles(
        audios,
        input_dir,
        lambda audio_path: whisper.load_model(model_name).transcribe(
            audio_path, task=task, language=language
        ),
    )

    if srt_only:
        return

    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}_result.mp4")
        print(f"Cropping video to TikTok format: {filename(path)}...")

        # Create a temporary cropped video file
        cropped_path = os.path.join(output_dir, f"{filename(path)}_cropped.mp4")

        # Crop video to TikTok format
        crop_and_add_overlay(path, cropped_path, face_cascade)

        print(f"Adding subtitles to {filename(path)}...")

        ffmpeg.input(cropped_path).output(
            out_path,
            vf=f"subtitles={srt_path}:force_style='Fontname=Candyz,Fontsize=26,PrimaryColour=&H00FFFFFF,OutlineColour=&H00FF2DA1'",
            acodec="copy",
        ).run(overwrite_output=True)

        # Remove temporary cropped video file
        os.remove(cropped_path)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="directory containing the input video files",
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=whisper.available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--output_srt",
        action="store_true",
        help="whether to output the .srt file along with the video files",
    )
    parser.add_argument(
        "--srt_only",
        action="store_true",
        help="only generate the .srt file and not create overlayed video",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="whether to print out the progress and debug messages",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="What is the origin language of the video? If unset, it is detected automatically.",
    )

    args = parser.parse_args()

    # Load Haar cascades only once
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    process_videos(
        args.input_dir,
        args.model,
        args.output_srt,
        args.srt_only,
        args.language,
        args.task,
        face_cascade,
    )


if __name__ == "__main__":
    main()
