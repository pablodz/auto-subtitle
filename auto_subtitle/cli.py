import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from .utils import filename, write_srt
import cv2


def crop_and_add_overlay(input_path, output_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Read the input video
    cap = cv2.VideoCapture(input_path)
    success, frame = cap.read()
    if not success:
        print("Failed to read input video.")
        return False

    # Detect faces in the first frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        print("No faces found in the input video.")
        return False

    print(f"Found {len(faces)} faces in the input video.")

    # Get the biggest face based on area
    biggest_face = max(faces, key=lambda f: f[2] * f[3])

    # Draw rectangle on the biggest face
    (x, y, w, h) = biggest_face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # save on generated folder
    face_output_path = os.path.join(output_path, "biggest_face.jpg")
    face_image = frame[y : y + h, x : x + w]
    cv2.imwrite(face_output_path, face_image)

    # Adjust the size of the bounding box to provide a more zoomed-in effect
    zoom_factor = 0.5
    x -= int(zoom_factor * w)
    y -= int(zoom_factor * h)
    w = int((1 + 2 * zoom_factor) * w)
    h = int((1 + 2 * zoom_factor) * h)

    # Calculate the position and size of the webcam video
    webcam_x = x
    webcam_y = y
    webcam_width = w
    webcam_height = h

    # Crop the original video to obtain the webcam video
    webcam_video = ffmpeg.input(input_path).crop(
        webcam_width, webcam_height, webcam_x, webcam_y
    )

    # Crop the input video to the middle
    cropped_video = ffmpeg.input(input_path).filter("crop", "ih*(9/16)", "ih")

    # Overlay the webcam video on top of the cropped video
    output_video = cropped_video.overlay(webcam_video, x=0, y=0)

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


def get_subtitles(audio_paths, output_srt, output_dir, transcribe):
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")

        print(f"Generating subtitles for {filename(path)}... This might take a while.")

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)

        subtitles_path[path] = srt_path

    return subtitles_path


def process_videos(
    input_dir, model_name, output_dir, output_srt, srt_only, language, task
):
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection."
        )
        language = "en"
    elif language != "auto":
        language = language

    model = whisper.load_model(model_name)
    videos = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.lower().endswith(".mp4")
    ]
    audios = get_audio(videos)
    subtitles = get_subtitles(
        audios,
        output_srt or srt_only,
        output_dir,
        lambda audio_path: model.transcribe(audio_path, task=task, language=language),
    )

    if srt_only:
        return

    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

        print(f"Cropping video to TikTok format: {filename(path)}...")

        # Create a temporary cropped video file
        cropped_path = os.path.join(output_dir, f"{filename(path)}_cropped.mp4")

        # Crop video to TikTok format
        crop_and_add_overlay(path, cropped_path)

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

    process_videos(
        args.input_dir,
        args.model,
        "./generated",
        args.output_srt,
        args.srt_only,
        args.language,
        args.task,
    )


if __name__ == "__main__":
    main()
