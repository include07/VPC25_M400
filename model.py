#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################



def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`, 
        which ensures compatibility with `soundfile.write()`.
    sr : int
        The sample rate of the processed audio.
    """
    import noisereduce as nr
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import soundfile as sf
    import os
    import random

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    # Read the source audio file
    def transcribe_audio(file_paths):
        """
        Transcribes one or more audio files using the Whisper model.

        Args:
            file_paths: A string (single file path) or a list of strings (multiple file paths).

        Returns:
            A dictionary mapping file path to its transcribed text.
        """
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        import soundfile as sf
        import noisereduce as nr
        import os

        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3-turbo"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        results = {}

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            try:
                audio_array, sample_rate = sf.read(file_path)
                
                if audio_array.ndim > 1:  # Convert stereo to mono
                    audio_array = audio_array.mean(axis=1)
                #filter for noise
                audio_array = nr.reduce_noise(y=audio_array, sr=sample_rate)
                audio_data = {"array": audio_array, "sampling_rate": sample_rate}
                generate_kwargs = {
                #     "max_new_tokens": 448,
                    "language": "english",
                     "num_beams": 2,
                #     "condition_on_prev_tokens": False,
                #     "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
                #     "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    "return_timestamps": True,
                }
                result = pipe(audio_data, generate_kwargs = generate_kwargs)
                results[file_path] = result["text"]
                print(f"Transcription for {file_path}: {result['text']}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results[file_path] = None

        return results
    # Apply your anonymization algorithm
    def anonymize_audio(file_path):
        """
        Given a file path to an audio file, returns the anonymized audio and its sampling rate.
        The process is:
        1. Transcribe the audio file.
        2. Synthesize anonymized speech using the transcribed text.
        
        Args:
            file_path: Path to the input audio file.
             
        Returns:
            (audio_array, sampling_rate): A tuple where audio_array is the anonymized audio data 
                                        and sampling_rate is the sampling rate (26000 in this case).
        """
        # Transcribe original audio file
        transcription_dict = transcribe_audio(file_path)
        original_text = transcription_dict.get(file_path)
        
        if not original_text:
            raise ValueError("Transcription failed. Cannot anonymize the file.")

        # Use KPipeline (from the kokoro package) to generate anonymized audio.
        from kokoro import KPipeline
        # Instantiate the pipeline and generate audio.
        pipeline_k = KPipeline(lang_code='a')  # Use American English; adjust as needed.
        
        # Synthesize anonymized audio using the transcribed text.
        # The split_pattern will separate chunks, but with a single sentence it returns one chunk.

        generator = pipeline_k([original_text], voice="af_heart", speed=1, split_pattern=r'\n+')
        
        # Combine audio chunks if several are returned.
        anonymized_audio = []
        for _, _, audio in generator:
            anonymized_audio.append(audio)
        
        # Concatenate if multiple chunks were generated.
        if len(anonymized_audio) == 0:
            raise ValueError("No audio generated from the anonymization pipeline.")
        elif len(anonymized_audio) == 1:
            final_audio = anonymized_audio[0]
        else:
            # using numpy to concatenate if more than one segment is present.
            import numpy as np
            final_audio = np.concatenate(anonymized_audio)

        sampling_rate = 26000  # The sampling rate is defined in the KPipeline generation.
        
        # Optionally, save the anonymized audio to a file.
        # output_filename = "anonymized.wav"
        # sf.write(output_filename, final_audio, sampling_rate)
        # print(f"Anonymized audio saved as {output_filename}")
        final_audio = nr.reduce_noise(y=final_audio, sr=sampling_rate)
        return final_audio, sampling_rate
    # Output:
    final_audio, sr = anonymize_audio(input_audio_path)
    return final_audio, sr