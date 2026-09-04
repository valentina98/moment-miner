import functools


@functools.lru_cache(maxsize=1)
def _model(model_size: str, device: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type="auto")


def transcribe(
    path: str, model_size: str = "small", device: str = "auto"
) -> list[tuple[float, float, str]]:
    segments, _info = _model(model_size, device).transcribe(
        path, vad_filter=True, condition_on_previous_text=False
    )
    # Whisper hallucinates captions on music/ambient sound; keep them out
    # of the searchable transcript.
    return [
        (s.start, s.end, s.text)
        for s in segments
        if s.no_speech_prob < 0.66 and s.avg_logprob > -1.0
    ]
