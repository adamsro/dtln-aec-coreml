#!/bin/bash
# Record real-world echo samples by playing lpb files through speakers
# while recording from microphone

SAMPLES_DIR="audio_samples"
DURATION_EXTRA=1  # Extra seconds to record after playback

for lpb_file in "$SAMPLES_DIR"/*_lpb.wav; do
    if [[ -f "$lpb_file" ]]; then
        base=$(basename "$lpb_file" _lpb.wav)
        output_file="$SAMPLES_DIR/${base}_realworld_mic.wav"
        
        # Get duration of lpb file
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$lpb_file")
        record_duration=$(echo "$duration + $DURATION_EXTRA" | bc)
        
        echo "Recording: $base"
        echo "  Playing: $lpb_file (${duration}s)"
        echo "  Recording to: $output_file (${record_duration}s)"
        echo "  >>> Press Enter to start, Ctrl+C to skip <<<"
        read
        
        # Start recording in background
        ffmpeg -y -f avfoundation -i ":0" -t "$record_duration" -ar 16000 -ac 1 -acodec pcm_f32le "$output_file" 2>/dev/null &
        RECORD_PID=$!
        
        # Small delay to ensure recording starts
        sleep 0.3
        
        # Play the lpb file
        afplay "$lpb_file"
        
        # Wait for recording to finish
        wait $RECORD_PID
        
        echo "  Saved: $output_file"
        echo ""
    fi
done

echo "Done! Real-world samples saved."
