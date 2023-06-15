import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from .utils import filename, str2bool, write_srt


def crop_video(input_path, output_path):
    video = ffmpeg.input(input_path)
    audio = video.audio
    video = video.filter("crop", "ih*(9/16)", "ih")
    ffmpeg.output(video, audio, output_path, crf=21).run()

    return True


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

    model_name = args.model
    output_dir = "./generated"
    output_srt = args.output_srt
    srt_only = args.srt_only
    language = args.language
    input_dir = args.input_dir

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
        lambda audio_path: model.transcribe(
            audio_path, task=args.task, language=language
        ),
    )

    if srt_only:
        return

    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

        print(f"Cropping video to TikTok format: {filename(path)}...")

        # Create a temporary cropped video file
        cropped_path = os.path.join(output_dir, f"{filename(path)}_cropped.mp4")

        # Crop video to TikTok format
        crop_video(path, cropped_path)

        print(f"Adding subtitles to {filename(path)}...")

        ffmpeg.input(cropped_path).output(
            out_path,
            vf=f"subtitles={srt_path}:force_style='Fontname=Candyz,Fontsize=26,PrimaryColour=&H00FFFFFF,OutlineColour=&H00FF2DA1'",
            acodec="copy",
        ).run(overwrite_output=True)

        # Remove temporary cropped video file
        os.remove(cropped_path)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


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


def get_subtitles(
    audio_paths: list, output_srt: bool, output_dir: str, transcribe: callable
):
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


if __name__ == "__main__":
    main()
