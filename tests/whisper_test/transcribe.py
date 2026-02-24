from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    "input2.wav",
    language="it"
)

print("Lingua rilevata:", info.language)

with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(f"Lingua rilevata: {info.language}\n")
    for segment in segments:
        f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
