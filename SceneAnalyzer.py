import cv2
import os
import base64
import requests
from moviepy.editor import VideoFileClip
import logging
import json
import sys
import threading
logging.getLogger('moviepy').setLevel(logging.ERROR)
import time
from functools import wraps
from dotenv import load_dotenv
#final_arr=[]
load_dotenv()

speech_key=os.environ["AZURE_SPEECH_KEY"]

#azure whisper key *
AZ_WHISPER=os.environ["AZURE_WHISPER_KEY"]

#Azure whisper deployment name *
azure_whisper_deployment=os.environ["AZURE_WHISPER_DEPLOYMENT"]

#Azure whisper endpoint (just name) *
azure_whisper_endpoint=os.environ["AZURE_WHISPER_ENDPOINT"]

#azure openai vision api key *
azure_vision_key=os.environ["AZURE_VISION_KEY"]

#Audio API type (OpenAI, Azure)*
audio_api_type=os.environ["AUDIO_API_TYPE"]

#GPT4 vision APi type (OpenAI, Azure)*
vision_api_type=os.environ["VISION_API_TYPE"]

#OpenAI API Key*
openai_api_key=os.environ["OPENAI_API_KEY"]

#GPT4 Azure vision API Deployment Name*
vision_deployment=os.environ["VISION_DEPLOYMENT_NAME"]

#GPT
vision_endpoint=os.environ["VISION_ENDPOINT"]

def log_execution_time(func):
    @wraps(func)  # Preserves the name and docstring of the decorated function
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete.")
        return result
    return wrapper




class Spinner:
    def __init__(self, message="Processing..."):
        self.spinner_symbols = "|/-\\"
        self.idx = 0
        self.message = message
        self.stop_spinner = False

    def spinner_task(self):
        while not self.stop_spinner:
            sys.stdout.write(f"\r{self.message} {self.spinner_symbols[self.idx % len(self.spinner_symbols)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            self.idx += 1

    def start(self):
        self.stop_spinner = False
        self.thread = threading.Thread(target=self.spinner_task)
        self.thread.start()

    def stop(self):
        self.stop_spinner = True
        self.thread.join()
        sys.stdout.write('\r' + ' '*(len(self.message)+2) + '\r')  # Erase spinner
        sys.stdout.flush()
chapter_summary = {}

@log_execution_time
def AnalyzeVideo(vp,fi,fpi):
# Constants
    video_path = vp  # Replace with your video path
    output_frame_dir = 'frames'
    output_audio_dir = 'audio'
    transcriptions_dir = 'transcriptions'
    frame_interval = fi  # seconds 180
    frames_per_interval = fpi

    # Ensure output directories exist
    for directory in [output_frame_dir, output_audio_dir, transcriptions_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Encode image to base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def send_post_request(resource_name, deployment_name, api_key,data):
            url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2023-12-01-preview"
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key
            }

            response = requests.post(url, headers=headers, data=json.dumps(data))
            return response
    # GPT-4 vision analysis function
    def gpt4_vision_analysis(image_path, api_key, summary, trans):
        cont=[
                {
                    "type": "text",
                    "text": f"Current Summary up to last {frame_interval} seconds: "+summary
                },
                {
                    "type": "text",
                    "text": f"Audio Transcription for last {frame_interval} seconds: "+trans
                },
                {
                    "type": "text",
                    "text": f"Next are the {frames_per_interval} frames from the last {frame_interval} seconds of the video:"
                }

                ]
        for img in image_path:
            base64_image = encode_image(img)
            cont.append( {
                        "type": "text",
                        "text": f"Below this is {img} (s is for seconds). use this to provide timestamps and understand time"
                    })
            cont.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })


        json_form=str([json.dumps({"title":"Chapter 1: A new beginning","start_frame":"0.0s","end_frame":"253.55s","scenes":[{"title":"Scene 1: it started","description":"The thing happened","start_frame":"0.0s","end_frame":"30.0s"},{"title":"Scene 2: around again","description":"Another thing happened","start_frame":"30.0s","end_frame":"75.0s"}]}),json.dumps({"title":"Chapter 2: Next steps","start_frame":"253.55s","end_frame":"604.90s","scenes":[{"title":"Scene 1: new hope","description":"The thing happened","start_frame":"275.0s","end_frame":"310.0s"},{"title":"Scene 2: bad days","description":"Another thing happened","start_frame":"310.0s","end_frame":"360.0s"}]})])
        if(vision_api_type=="Azure"):
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            payload2 = {

                "messages": [
                    {
                    "role": "system",
                    "content": f"""You are VideoAnalyzerGPT. Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
                    as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
                    You are then provided a Current Chapter Breakdown of the video so far (3 most recent chapters only),
                    which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until iteratively we have a full breakdown of all the chapters of the video.
                    As the main intelligence of this system, you are responsible for building the Current Chapter Breakdown using both the audio you are being provided
                    via transcription, as well as the image of the frame.
                    Always and only return as your output the updated Current Chapter Breakdown in format ```{json_form}```.
                    (the format is a template, make sure to start at chapter 1 in your generation if there is not one already.)
                    The start and end frames represent the times that a scene and chapter starts and ends, use the data provided above each image, and in the audio to service this feature.
                    You can think through your responses step by step. Determine the Chapters contextually using the audio and analyzed video frames.
                    You dont need to provide a new chapter for every frame, the chapters should represent overarching themes and moments.
                    Always provide new or updated chapters in your response, and consider them all for editing purposes on each pass,
                    the Chapter Response Should be a JSON object array, with each chapter being a json object, with each key being a scene title in the chapter,
                    with the value being an array of information about the scene, with the first key in each object being the title of the chapter.
                    The thresholds required for a new chapter are: Major Thematic Change, Major Story Change, Major Setting Change.
                    Think through your Chapter assignment process step by step before providing the response JSON.
                    Do not make up timestamps, use the ones provided with each frame.
                    Provide back the response as JSON, and always and only return back JSON following the format specified.
                    Scenes in a given chapter must be contiguous
                    """

                    },
                {
                    "role": "user",
                    "content": cont
                }
                ],
                "max_tokens": 4000,
                "seed": 42


            }
            response=send_post_request(vision_endpoint,vision_deployment,azure_vision_key,payload2)
        else:
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                    "role": "system",
                    "content": f"""You are VideoAnalyzerGPT. Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
                    as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
                    You are then provided a Current Chapter Breakdown of the video so far (3 most recent chapters only),
                    which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until iteratively we have a full breakdown of all the chapters of the video.
                    As the main intelligence of this system, you are responsible for building the Current Chapter Breakdown using both the audio you are being provided
                    via transcription, as well as the image of the frame.
                    Always and only return as your output the updated Current Chapter Breakdown in format ```{json_form}```.
                    (the format is a template, make sure to start at chapter 1 in your generation if there is not one already.)
                    The start and end frames represent the times that a scene and chapter starts and ends, use the data provided above each image, and in the audio to service this feature.
                    You can think through your responses step by step. Determine the Chapters contextually using the audio and analyzed video frames.
                    You dont need to provide a new chapter for every frame, the chapters should represent overarching themes and moments.
                    Always provide new or updated chapters in your response, and consider them all for editing purposes on each pass,
                    the Chapter Response Should be a JSON object array, with each chapter being a json object, with each key being a scene title in the chapter,
                    with the value being an array of information about the scene, with the first key in each object being the title of the chapter.
                    The thresholds required for a new chapter are: Major Thematic Change, Major Story Change, Major Setting Change.
                    Think through your Chapter assignment process step by step before providing the response JSON.
                    Do not make up timestamps, use the ones provided with each frame.
                    Provide back the response as JSON, and always and only return back JSON following the format specified.
                    Scenes in a given chapter must be contiguous
                    """                    },
                {
                    "role": "user",
                    "content": cont
                }
                ],
                "max_tokens": 4000,
                "seed": 42,


            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()



    def update_chapter_summary(new_json_string):
        global chapter_summary
        if new_json_string.startswith('json'):
        # Remove the first occurrence of 'json' from the response text
            new_json_string = new_json_string[4:]
        else:
            new_json_string = new_json_string
        # Assuming new_json_string is the JSON format string returned from your API call
        new_chapters_list = json.loads(new_json_string)

        # Iterate over the list of new chapters
        for chapter in new_chapters_list:
            chapter_title = chapter['title']
            # Update the chapter_summary with the new chapter
            chapter_summary[chapter_title] = chapter

        # Get keys of the last three chapters
        last_three_keys = list(chapter_summary.keys())[-3:]
        # Get the last three chapters as an array
        last_three_chapters = [chapter_summary[key] for key in last_three_keys]

        return last_three_chapters

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second



    # Process video
    current_frame = 0
    current_second = 0
    current_summary=""
    packet=[]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    # Constants


    # Load video audio
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration  # Duration of the video in seconds

    # Process video
    current_frame = 0  # Current frame initialized to 0
    current_second = 0  # Current second initialized to 0
    current_summary=""
    packet=[]
    current_interval_start_second = 0
    capture_interval_in_frames = int(fps * frame_interval / frames_per_interval)  # Interval in frames to capture the image
    spinner = Spinner("Capturing Video and Audio...")
    spinner.start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        current_second = current_frame / fps

        if current_frame % capture_interval_in_frames == 0 and current_frame != 0:
            # Extract and save frame
            # Save frame at the exact intervals
            frame_name = f'frame_at_{current_second}s.jpg'
            #print(frame_name)
            frame_path = os.path.join(output_frame_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            packet.append(frame_path)
        if len(packet) == frames_per_interval or (current_interval_start_second + frame_interval) < current_second:
            audio_name = f'audio_at_{current_interval_start_second}s.mp3'
            audio_path = os.path.join(output_audio_dir, audio_name)
            audio_clip = video_clip.subclip(current_interval_start_second, min(current_interval_start_second + frame_interval, video_clip.duration))  # Avoid going past the video duration
            audio_clip.audio.write_audiofile(audio_path, codec='mp3', verbose=False, logger=None)
            #print(f'Extracted audio and frames from {current_interval_start_second} to {min(current_interval_start_second + frame_interval, video_clip.duration)} second.\n')


            # TODO: Add code for transcribing audio with OpenAI Whisper API (as shown in previous examples)
            spinner.stop()
            spinner = Spinner("Transcribing Audio...")
            spinner.start()
            def transcribe_audio(audio_path, endpoint, api_key, deployment_name):
                # url = f"{endpoint}/openai/deployments/{deployment_name}/audio/transcriptions?api-version=2024-02-01"
                url = f"https://aoai-sweden-minggu.openai.azure.com/openai/deployments/whisper/audio/transcriptions?api-version=2024-02-01"
                print(f"url: {url}")
                print(f"api-key: {api_key}")
                headers = {
                    "api-key": api_key,
                    "Content-Type": "multipart/form-data"
                }
                json = {
                    "file": (audio_path.split("/")[-1], open(audio_path, "rb"), "audio/mp3"),
                    "locale": "en-US",
                }
                response = requests.post(url, headers=headers, files=json)

                return response

            if(audio_api_type=="Azure"):
                response = transcribe_audio(audio_path,azure_whisper_endpoint,AZ_WHISPER,azure_whisper_deployment)
            else:
                from openai import OpenAI
                client = OpenAI()

                audio_file= open(audio_path, "rb")
                response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
                )

            current_transcription=""
            print(f"response: {response}")
            for item in response.segments:
                current_transcription+=str(round(item["start"],2))+"s - "+str(round(item["end"],2))+"s: "+item["text"]+"\n"

            spinner.stop()
            spinner = Spinner("Processing Frames and Audio with AI...")
            spinner.start()

            # Simulate some long-running process


            # Analyze frame with GPT-4 vision
            vision_response = gpt4_vision_analysis(packet, openai_api_key, current_summary, current_transcription)
            try:
                vision_analysis = vision_response["choices"][0]["message"]["content"]
            except:
                print(vision_response)
            try:
                chapter_text = str(vision_analysis).split("```")[1]
                last_three_chapters = update_chapter_summary(chapter_text)
                # Convert the last three chapters back to JSON string to update current_summary
                current_summary = json.dumps(last_three_chapters, ensure_ascii=False)
            except Exception as e:
                print("bad json",str(e))
                current_summary=str(vision_analysis)
            spinner.stop()
            print(f'{json.dumps(current_summary,indent=2)} \n')
            spinner = Spinner("Capturing Video and Audio...")
            spinner.start()
            packet.clear()  # Clear packet after analysis
            current_interval_start_second += frame_interval  # Move to the next set of frames

        if current_second >= video_clip.duration:
            break

        current_frame += 1
        current_second = current_frame / fps
        #current_second = int(current_frame / fps)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print('Extraction, analysis, and transcription completed.')
    print("\n\n\n"+json.dumps(chapter_summary,indent=2))

AnalyzeVideo("207566398_test_video.mp4",60,10)
with open('chapterBreakdown.json', 'w') as f:
    # Write the data to the file in JSON format
    json.dump(chapter_summary, f, indent=4)