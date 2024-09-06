import cv2
import os
import base64
import requests
from moviepy.editor import VideoFileClip
import logging
import json
import sys
import openai
import threading
from retrying import retry
logging.getLogger('moviepy').setLevel(logging.ERROR)
import time
from functools import wraps
from dotenv import load_dotenv
import time

final_arr=[]
load_dotenv()

#azure speech key
speech_key=os.environ["AZURE_SPEECH_KEY"]

#azure whisper key *
AZ_WHISPER=os.environ["AZURE_WHISPER_KEY"]

#Azure whisper deployment name *
azure_whisper_deployment=os.environ["AZURE_WHISPER_DEPLOYMENT"]

#Azure whisper endpoint (just name) *
azure_whisper_endpoint=os.environ["AZURE_WHISPER_ENDPOINT"]

#azure openai vision api key *
#azure_vision_key=os.environ["AZURE_VISION_KEY"]

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
miss_arr=[]

@log_execution_time
def AnalyzeVideo(vp,fi,fpi,face_rec=False):
# Constants
    video_path = vp  # Replace with your video path
    output_frame_dir = 'frames'
    output_audio_dir = 'audio'
    global_transcript=""
    transcriptions_dir = 'transcriptions'
    frame_interval = fi  # seconds 180
    frames_per_interval = fpi
    totalData=""
    # Ensure output directories exist
    for directory in [output_frame_dir, output_audio_dir, transcriptions_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Encode image to base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    #GPT 4 Vision Azure helper function
    def send_post_request(resource_name, deployment_name, api_key,data):
        url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2023-12-01-preview"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }

        print(f"resource_name: {resource_name}")
        print(f"Sending POST request to {url}")
        print(f"Headers: {headers}")
        # print(f"Data: {json.dumps(data)}")
        print(f"api_key: {api_key}")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response
    # GPT-4 vision analysis function
    @retry(stop_max_attempt_number=3)
    def gpt4_vision_analysis(image_path, api_key, summary, trans):
        #array of metadata
        cont=[

                {
                    "type": "text",
                    "text": f"Audio Transcription for last {frame_interval} seconds: "+trans
                },
                {
                    "type": "text",
                    "text": f"Next are the {frames_per_interval} frames from the last {frame_interval} seconds of the video:"
                }

                ]
        #adds images and metadata to package
        for img in image_path:
            base64_image = encode_image(img)
            cont.append( {
                        "type": "text",
                        "text": f"Below this is {img} (s is for seconds). use this to provide timestamps and understand time"
                    })
            cont.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail":"high"
                        }
                    })

        #extraction template
        json_form=str([json.dumps({"Start_Timestamp":"4.97s","sentiment":"Positive, Negative, or Neutral","End_Timestamp":"16s","scene_theme":"Dramatic","characters":"Man in hat, woman in jacket","summary":"Summary of what is occuring around this timestamp with actions included, uses both transcript and frames to create full picture, be detailed and attentive, be serious and straightforward in your description.","actions":"Actions extracted via frame analysis","key_objects":"Any objects in the timerange, include colors along with descriptions. all people should be in this, with as much detail as possible extracted from the frame (clothing,colors,age). Be incredibly detailed"}),
                       json.dumps({"Start_Timestamp":"16s","sentiment":"Positive, Negative, or Neutral","End_Timestamp":"120s","scene_theme":"Emotional, Heartfelt","characters":"Man in hat, woman in jacket","summary":"Summary of what is occuring around this timestamp with actions included, uses both transcript and frames to create full picture, detailed and attentive, be serious and straightforward in your description.","actions":"Actions extracted via frame analysis","key_objects":"Any objects in the timerange, include colors along with descriptions. all people should be in this, all people should be in this, with as much detail as possible extracted from the frame (clothing,colors,age). Be incredibly detailed"})])
        
        if(vision_api_type=="Azure"):
            print("sending request to gpt-4o")
            payload2 = {

                "messages": [
                    {
                    "role": "system",
                    "content": f"""You are VideoAnalyzerGPT. Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
                    as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
                    You are then to generate and provide a Current Action Summary (An Action summary of the video,
                    with actions being added to the summary via the analysis of the frames) of the portion of the video you are considering ({frames_per_interval}
                    frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until we have a full action summary of the portion of the video you are considering,
                    that includes all actions taken by characters in the video as extracted from the frames. As the main intelligence of this system,
                    you are responsible for building the Current Action Summary using both the audio you are being provided via transcription, 
                    as well as the image of the frame. Always and only return as your output the updated Current Action Summary in format ```{json_form}```. 
                    Do not make up timestamps, use the ones provided with each frame. 
                    Construct each action summary block from mulitple frames, each block should represent a scene or distinct trick or move in the video, minimum of 2 blocks per output.
                    Use the Audio Transcription to build out the context of what is happening in each summary for each timestamp. 
                    Consider all frames and audio given to you to build the Action Summary. Be as descriptive and detailed as possible, 
                    Make sure to try and Analyze the frames as a cohesive 10 seconds of video.
gr

                    You are analyzing a FIFA football game, so use sports terminology and analyze from the perspective of a sports announcer. The list of players and there numbers are provided below, use this to correctly identify which players take what action by their number in the vieo.
                    13 Vicario
                    23 Porro
                    17 Romero
                    37 Van de Ven
                    33 Davies
                    5 Højbjerg
                    30 Bentancur
                    21 Kulusevski
                    10 Maddison
                    16 Werner
                    7 Son
                    19 Trossard
                    29 Havertz
                    7 Saka
                    41 Rice
                    5 Partey

                You are an in car AI assistant to help passengers create comfortable environment. 
                            If you find the baby sleeping image frames, you should call function of turning down music volume. 
                            If you find images where children are sticking their heads or hands out of the window, you should call function of 
                            educating the children with a command and issue a warning. 
                            Show the image number that triggered either one of the two function calling.

                    your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON"""
                    },
                {
                    "role": "user",
                    "content": cont
                }
                ],
                "max_tokens": 4000,
                "seed": 42


            }
            response=send_post_request(vision_endpoint,vision_deployment,openai_api_key,payload2)

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
                    You are then to generate and provide a Current Action Summary (An Action summary of the video,
                    with actions being added to the summary via the analysis of the frames) of the portion of the video you are considering ({frames_per_interval}
                    frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until we have a full action summary of the portion of the video you are considering,
                    that includes all actions taken by characters in the video as extracted from the frames. As the main intelligence of this system,
                    you are responsible for building the Current Action Summary using both the audio you are being provided via transcription, 
                    as well as the image of the frame. Always and only return as your output the updated Current Action Summary in format ```{json_form}```. 
                    Do not make up timestamps, use the ones provided with each frame. 
                    Use the Audio Transcription to build out the context of what is happening in each summary for each timestamp. 
                    Consider all frames and audio given to you to build the Action Summary. Be as descriptive and detailed as possible, 
                    your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON"""
                },
            {
                "role": "user",
                "content": cont
            }
            ],
            "max_tokens": 4000,
            "seed": 42


            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        if(response.status_code!=200):
            return -1
        else:
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

    packet_count=1
    # Initialize known face encodings and their names if provided
    known_face_encodings = []
    known_face_names = []

    def load_known_faces(known_faces):
        for face in known_faces:
            image = face_recognition.load_image_file(face['image_path'])
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(face['name'])

    # Call this function if you have known faces to match against
    # load_known_faces(array_of_recognized_faces)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        current_second = current_frame / fps

        if current_frame % capture_interval_in_frames == 0 and current_frame != 0:
            print(f"BEEP {current_frame}")
            # Extract and save frame
            # Save frame at the exact intervals
            if(face_rec==True):
                import face_recognition
                import numpy
                ##small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                #rgb_frame = frame[:, :, ::-1]  # Convert the image from BGR color (which OpenCV uses) to RGB color  
                rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])
                face_locations = face_recognition.face_locations(rgb_frame)
                #print(face_locations)
                face_encodings=False
                if(len(face_locations)>0):

                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    print(face_encodings)

                # Initialize an array to hold names for the current frame
                face_names = []
                if(face_encodings!=False):
                    for face_encoding in face_encodings:  
                        # See if the face is a match for the known faces  
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.4)  
                        name = "Unknown"  
            
                        # If a match was found in known_face_encodings, use the known person's name.  
                        if True in matches:  
                            first_match_index = matches.index(True)  
                            name = known_face_names[first_match_index]  
                        else:  
                            # If no match and we haven't assigned a name, give a new name based on the number of unknown faces  
                            name = f"Person {chr(len(known_face_encodings) + 65)}"  # Starts from 'A', 'B', ...  
                            # Add the new face to our known faces  
                            known_face_encodings.append(face_encoding)  
                            known_face_names.append(name)  
            
                        face_names.append(name)  
            
                    # Draw rectangles around each face and label them  
                    for (top, right, bottom, left), name in zip(face_locations, face_names):  
                        # Draw a box around the face  
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  
            
                        # Draw a label with a name below the face  
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX  
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)  
            
                # Save the frame with bounding boxes  
                frame_name = f'frame_at_{current_second}s.jpg'  
                frame_path = os.path.join(output_frame_dir, frame_name)  
                cv2.imwrite(frame_path, frame)
            else:
                frame_name = f'frame_at_{current_second}s.jpg'
                #print(frame_name)
                frame_path = os.path.join(output_frame_dir, frame_name)
                cv2.imwrite(frame_path, frame)
            packet.append(frame_path)
        #if packet len is appropriate (meaning FPI is hit) then process the audio for transcription
        if len(packet) == frames_per_interval or (current_interval_start_second + frame_interval) < current_second:
            current_transcription=""
            if video_clip.audio is not None:
                audio_name = f'audio_at_{current_interval_start_second}s.mp3'
                audio_path = os.path.join(output_audio_dir, audio_name)
                audio_clip = video_clip.subclip(current_interval_start_second, min(current_interval_start_second + frame_interval, video_clip.duration))  # Avoid going past the video duration
                audio_clip.audio.write_audiofile(audio_path, codec='mp3', verbose=False, logger=None)

                headers = {
                    'Authorization': f'Bearer {openai_api_key}',
                }
                files = {
                    'file': open(audio_path, 'rb'),
                    'model': (None, 'whisper-1'),
                }
                spinner.stop()
                spinner = Spinner("Transcribing Audio...")
                spinner.start()
                # Actual audio transcription occurs in either OpenAI or Azure
                @retry(stop_max_attempt_number=3)
                def transcribe_audio(audio_path, endpoint, api_key, deployment_name):
                    url = f"{endpoint}/openai/deployments/{deployment_name}/audio/transcriptions?api-version=2023-09-01-preview"

                    headers = {
                        "api-key": api_key,
                    }
                    json = {
                        "file": (audio_path.split("/")[-1], open(audio_path, "rb"), "audio/mp3"),
                    }
                    data = {
                        'response_format': (None, 'verbose_json')
                    }
                    response = requests.post(url, headers=headers, files=json, data=data)

                    return response

                if(audio_api_type == "Azure"):
                    response = transcribe_audio(audio_path, azure_whisper_endpoint, AZ_WHISPER, azure_whisper_deployment)
                else:
                    from openai import OpenAI
                    client = OpenAI()

                    audio_file = open(audio_path, "rb")
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                    )

                current_transcription = ""
                tscribe = ""
                # Process transcription response
                if(audio_api_type == "Azure"):
                    try:
                        for item in response.json()["segments"]:
                            tscribe += str(round(item["start"], 2)) + "s - " + str(round(item["end"], 2)) + "s: " + item["text"] + "\n"
                    except:
                        tscribe += ""
                else:
                    for item in response.segments:
                        tscribe += str(round(item["start"], 2)) + "s - " + str(round(item["end"], 2)) + "s: " + item["text"] + "\n"
                global_transcript += "\n"
                global_transcript += tscribe
                current_transcription = tscribe
            else:
                print("No audio track found in video clip. Skipping audio extraction and transcription.")
            spinner.stop()
            spinner = Spinner("Processing the "+str(packet_count)+" Frames and Audio with AI...")
            spinner.start()

            # Analyze frames with GPT-4 vision
            vision_response = gpt4_vision_analysis(packet, openai_api_key, current_summary, current_transcription)
            if(vision_response==-1):
                packet.clear()  # Clear packet after analysis
                current_interval_start_second += frame_interval
                current_frame += 1
                current_second = current_frame / fps
                continue
            time.sleep(5)
            try:
                vision_analysis = vision_response["choices"][0]["message"]["content"]
            except:
                print(vision_response)
            try:
                current_summary = vision_analysis
            except Exception as e:
                print("bad json",str(e))
                current_summary=str(vision_analysis)
            #print(current_summary)
            totalData+="\n"+str(current_summary)
            try:
                #print(vision_analysis)
                #print(vision_analysis.replace("'",""))
                vault=vision_analysis.split("```")
                if(len(vault)>1):
                    vault=vault[1]
                else:
                    vault=vault[0]
                vault=vision_analysis.replace("'","")
                vault=vault.replace("json","")
                vault=vault.replace("```","")

                if vault.startswith('json'):
                # Remove the first occurrence of 'json' from the response text
                    vault = vault[4:]
                    #print(vault)
                else:
                    vault = vault
                #print(vision_analysis.replace("'",""))
                data=json.loads(vault)
                #print(data)
                for item in data:
                    final_arr.append(item)
                    ##socket.emit('actionsummary', {'data': item}, namespace='/test')
                    with open('actionSummary.json', 'w') as f:
                    # Write the data to the file in JSON format
                        json.dump(final_arr, f, indent=4)
                    #print(item)
            except:
                miss_arr.append(vision_analysis)
                print("missed")

            spinner.stop()
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
    with open('actionSummary.json', 'w') as f:
    # Write the data to the file in JSON format
        json.dump(final_arr, f, indent=4)
        
    with open('transcript.txt', 'w') as f:
    # Write the data to the file in JSON format
        f.write(global_transcript)
    return final_arr

    #print("\n\n\n"+totalData)

#AnalyzeVideo("./medal.mp4",60,10)
AnalyzeVideo("test_video/car-driving.mov",6,10,False)
# print(miss_arr)
# if __name__ == "__main__": 
     

#     data=sys.argv[1].split(",")
#     print(data)
#     video_path = data[0] 
#     frame_interval = data[1]
#     frames_per_interval = data[2]  
#     face_rec=False
#     if(len(data)>3):
#         face_rec=True
  
#     AnalyzeVideo(video_path, int(frame_interval), int(frames_per_interval),face_rec) 
