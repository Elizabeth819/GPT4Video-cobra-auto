from dotenv import load_dotenv
from IPython.display import display, Image, Audio
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

import cv2  # We're using OpenCV to read video
import base64
import time
import io
import openai
import os
import requests

import streamlit as st
import tempfile
import numpy as np
from tempfile import NamedTemporaryFile
from PIL import Image
from moviepy.editor import VideoFileClip
import logging
import json
import sys
import threading
from retrying import retry
logging.getLogger('moviepy').setLevel(logging.ERROR)
import time
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

#speech_key=os.environ["AZURE_SPEECH_KEY"]
#speech_key="3b09c0cf3bc74cfdb4be8200fefeb35a"

#azure whisper key *
#AZ_WHISPER=os.environ["AZURE_WHISPER_KEY"]

#Azure whisper deployment name *
#azure_whisper_deployment=os.environ["AZURE_WHISPER_DEPLOYMENT"]

#Azure whisper endpoint (just name) *
#azure_whisper_endpoint=os.environ["AZURE_WHISPER_ENDPOINT"]

#azure openai vision api key *
os.environ["AZURE_VISION_KEY"] = "c7da7257d20b464e9dbd830ef2261304"
azure_vision_key="c7da7257d20b464e9dbd830ef2261304"

#Audio API type (OpenAI, Azure)*
#audio_api_type=os.environ["AUDIO_API_TYPE"]
audio_api_type="OpenAI"

#GPT4 vision APi type (OpenAI, Azure)*
os.environ["VISION_API_TYPE"] ="Azure"
vision_api_type="Azure"

#OpenAI API Key*
os.environ["OPENAI_API_KEY"] = "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"
openai_api_key="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"

#GPT4 Azure vision API Deployment Name*
#vision_deployment=os.environ["VISION_DEPLOYMENT_NAME"]
vision_deployment="ovv-vision-west"

#GPT
#vision_endpoint=os.environ["VISION_ENDPOINT"]
vision_endpoint="ovv-vision-west"

#"https://openaiswissrrd.openai.azure.com/openai/deployments/gpt4visionrrd/extensions/chat/completions?api-version=2023-07-01-preview"


load_dotenv()

#Utility functions



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

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response


#Load in the video and convert it to frames
import math
def process_raw_video_to_frames(video_path, output_frame_dir,fpi):

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    print(fps)
    sample_image_rate = math.floor(fps) * fpi # e.g. fpi = 2 means that every 60 frames we will capture 1 frame

    # Load video audio
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration  # Duration of the video in seconds
    print(video_duration)
    total_frame_count = (fps * video_duration)
    print(total_frame_count)

    # Process video
    current_frame = 1  # Current frame initialized to 1
    packet=[]
    start = time.time()

    packet_count=1
    ret = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("video is not opened")
            break
        else:
            elapsed = time.time() - start
            elapsed = elapsed
            #print(current_frame,sample_image_rate)
            if current_frame % sample_image_rate == 0:
                frame_name = f'frame_at_{elapsed}s.jpg'
                #frame_name = f'frame_at_{current_frame}.jpg'
                frame_path = os.path.join(output_frame_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                packet.append(frame_path)
            if current_frame>= total_frame_count:
                   break

        current_frame += 1
        time.sleep(1/fps)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return packet, fps, video_duration, total_frame_count





#Generate story based on frames with GPT-4v
def gpt4_vision_analysis(image_path, api_key,frame_count,video_duration,estimated_word_count):
        #array of metadata
        cont=[

                {
                    "type": "text",
                    "text": f"Next are the {frame_count} frames from the {video_duration} seconds of the video:"

                }

            ]
        #adds images and metadata to package
        try:
            for img in image_path:
                print(img)
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
        except Exception as e:
            print(e)
            raise Exception



        #extraction template
        # TODO: Should we use 'timerange' or 'time range' here?
        #json_form=str([json.dumps({"Beginning_Timestamp":"4.97s","Ending_Timestamp":"16s","Voice Over Description":"Description of what is occurring around the Beginning_Timestamp and Ending_Timestamp with actions included, frames will be used to create full description","actions":"Actions extracted via frame analysis","key_objects":"Any objects in the timerange, include colors along with descriptions. all people should be in this, with as much detail as possible extracted from the frame (clothing,colors,age)"})])
        #json_form=str([json.dumps({"Voice Over Description":"Description of what is occurring in the will be used to create full description"})])

        if(vision_api_type=="Azure"):
            payload2 = {

                "messages": [
                    {
                    "role": "system",
                    "content": f"You are Voice Over Agent. Your role is to take in as an input a video of {video_duration} seconds that contains {frame_count} frames split evenly throughout the video.The key goal of your operation, is to produce a descriptive video script that will be read out over the video to someone who is visually impaired. Describe in detail all of the people, objects and actions you see for the interval of frames and future frames that you have been supplied. Take note of image scene changes and image variation to introduce new descriptions of the scenes. Make sure the description is precise and matches the image for when it occurred and absolutely do not to exceed the estimated word count {estimated_word_count} for the overall voice over description. Do not add any timestamps to the output script. These are strict requirement that must be observed."
                    },
                {
                    "role": "user",
                    "content": cont
                }
                ],
                "max_tokens": 60,
                "seed": 42,
                "temperature": 0.1


            }
            #print(vision_endpoint)
            #print(vision_deployment)
            #print(azure_vision_key)
            #print(payload2)
            response=send_post_request(vision_endpoint,vision_deployment,azure_vision_key,payload2)

        else:
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            payload = {
            "model": "gpt-4-vision-preview",
            # "messages": [
            #     {
            #     "role": "system",
            #     "content": f"You are Voice Over Agent. Your job is to take in as an input video {video_duration} seconds long that contains {frame_count} frames split evenly throughout the video.The key goal of your operation, is to produce a descriptive video script that will be read out over the video to someone who is visually impaired.You are not to exceed {total_interval_words} words for the overall voice over script  nor are you to add any timestamps to the output these are strict requirement that must be observed."
            #         },
            #     },
            # {
            #     "role": "user",
            #     "content": cont
            # }
            # ],
                "messages": [
                    {
                    "role": "system",
                    "content": f"You are Voice Over Agent. Your role is to take in as an input a video of {video_duration} seconds that contains {frame_count} frames split evenly throughout the video.The key goal of your operation, is to produce a descriptive video script that will be read out over the video to someone who is visually impaired. Describe in detail all of the people, objects and actions you see for the interval of frames and future frames that you have been supplied. Take note of image scene changes and image variation to introduce new descriptions of the scenes. Make sure the description is precise and matches the image for when it occurred and absolutely do not to exceed the estimated word count {estimated_word_count} for the overall voice over description. Do not add any timestamps to the output script. These are strict requirement that must be observed. Use the timestamps provided above each image to fit your voiceover to match the timing of the video. It is very important that the voiceover generated lines up directly with the video, so that the voiceover content refers directly to what is on the screen. Refer to the timestamps to establish this relationship"
                    },
                {
                    "role": "user",
                    "content": cont
                }
                ],
                "max_tokens": 60,
                "seed": 42,
                "temperature": 0.1




            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if(response.status_code!=200):
            print(response.json())
            raise Exception
        else:
            return response.json()





#Actual audio transcription occurs in either OpenAI or Azure
def transcribe_audio(audio_path, endpoint, api_key, deployment_name):
        url = f"{endpoint}/openai/deployments/{deployment_name}/audio/transcriptions?api-version=2023-09-01-preview"
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
            response_format="text"
                )

        current_transcription=response
        return current_transcription

# Generate voice over from stories
def text_to_audio_sync_file_gen(text,audio_segment_filename):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": "onyx",
        },
    )

    audio_file_path = "./audio/" + audio_segment_filename + ".wav"
    with open(audio_file_path, "wb") as audio_file:
         for chunk in response.iter_content(chunk_size=1024 * 1024):
             audio_file.write(chunk)

    # To play the audio in Jupyter after saving
    Audio(audio_file_path)
    # Check if the request was successful

    if response.status_code != 200:
        print(response.json)
        raise Exception("Request failed with status code")
    # ...
    # Create an in-memory bytes buffer
    audio_bytes_io = io.BytesIO()

    # Write audio data to the in-memory bytes buffer
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio_bytes_io.write(chunk)

    # Important: Seek to the start of the BytesIO buffer before returning
    audio_bytes_io.seek(0)

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            tmpfile.write(chunk)
        audio_filename = tmpfile.name

    return audio_filename, audio_bytes_io




# Generate voice over from stories
def text_to_audio(text):
    print(text)
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": "onyx",
        },
    )

    audio_file_path = "output_audio.wav"
    with open(audio_file_path, "wb") as audio_file:
         for chunk in response.iter_content(chunk_size=1024 * 1024):
             audio_file.write(chunk)

    # To play the audio in Jupyter after saving
    Audio(audio_file_path)
    # Check if the request was successful

    if response.status_code != 200:
        print(response.json)
        raise Exception("Request failed with status code")
    # ...
    # Create an in-memory bytes buffer
    audio_bytes_io = io.BytesIO()

    # Write audio data to the in-memory bytes buffer
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio_bytes_io.write(chunk)

    # Important: Seek to the start of the BytesIO buffer before returning
    audio_bytes_io.seek(0)

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            tmpfile.write(chunk)
        audio_filename = tmpfile.name

    return audio_filename, audio_bytes_io




# Merge audio and video together
def merge_audio_video(video_filename, audio_filename, output_filename):
    print("Merging audio and video...")
    print("Video filename:", video_filename)
    print("Audio filename:", audio_filename)

    # Load the video file
    video_clip = VideoFileClip(video_filename)

    # Load the audio file
    audio_clip = AudioFileClip(audio_filename)

    # Set the audio of the video clip as the audio file
    final_clip = video_clip.set_audio(audio_clip)

    # Write the result to a file (without audio)
    final_clip.write_videofile(
        output_filename, codec='libx264', audio_codec='aac')

    # Close the clips
    video_clip.close()
    audio_clip.close()

    # Return the path to the new video file
    return output_filename





#Streamlit UI
def main():
    output_frame_dir = 'frames'
    output_audio_dir = 'audio'
    transcriptions_dir = 'transcriptions'
    fpi = 1


    # Ensure output directories exist
    for directory in [output_frame_dir, transcriptions_dir, output_audio_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    #Delete files in all directories

    #Delete files in the output_frame_dir
    for f in os.listdir(output_frame_dir):
        os.remove(os.path.join(output_frame_dir, f))

    #Delete files in the transcriptions dir
    for f in os.listdir(transcriptions_dir):
        os.remove(os.path.join(transcriptions_dir, f))

    #Testing without streamlit
    #sample_video = "gymnastics.mp4"

    packet = []
    narrative = ""

    st.set_page_config(page_title="Voice Over", page_icon=":bird:")

    st.header("Video voice over :bird:")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        st.video(uploaded_file)
        prompt = st.text_area(
            "Prompt", value="Click the Generate button below to create a short voiceover script that describes what is seen in this video.")


    if st.button('Generate', type="primary") and uploaded_file is not None:
        st.write("Processing video...")
        packet, fps, video_duration, total_frame_count = process_raw_video_to_frames(uploaded_file.name,output_frame_dir,fpi)
        #total_interval_words = int(video_duration * 2.0)
        est_word_count = int(video_duration * 1.6)
        est_word_count=15
        st.write("Total frames used in the voice over: ", len(packet))
        st.write("Total original video duration: ", video_duration)
        st.write("Total frame count: ", total_frame_count)
        st.write("Total fps: ", fps)
        st.write("Estimated word count (per interval) submitted to GPT4v: ", est_word_count)



        with st.spinner('Processing...'):
            if video_duration < 30:

                #Create the story for a short video i.e. less than 30 seconds
                #print("Short video")
                st.write("Processing short video")
                print(packet)
                vision_response = gpt4_vision_analysis(packet, azure_vision_key,total_frame_count,video_duration, est_word_count)

                vision_analysis = vision_response["choices"][0]["message"]["content"]
                #print(vision_analysis)
                #print(vision_analysis)
                #print(vision_analysis)
                st.write("Processing Text-to-Speech...")
                audio_file_name = text_to_audio(vision_analysis)
                #print(audio_file_name)
                st.write("Merging audio and video")
                #output_filename = merge_audio_video(uploaded_file.name,"output_audio.wav", "final_output.mp4")
                #st.video(output_filename)
            else:
                #Create the story for a longer video
                print("Longer video")
                chunked_frames_count = 0
                narrative = ""
                requests_per_call = 10
                frame_counter = 1
                call_counter = 1
                for frame in packet:
                    print ("Processing frame: ", frame_counter)
                    if frame_counter % requests_per_call == 0:
                        vision_response = gpt4_vision_analysis(packet[frame_counter - requests_per_call:requests_per_call* call_counter], azure_vision_key,10,10, 10)
                        print(vision_response)
                        vision_analysis = vision_response["choices"][0]["message"]["content"]
                        vision_analysis1 = vision_analysis
                        audio_filename, audio_bytes_io = text_to_audio_sync_file_gen(vision_analysis1, ("frame_interval"+ str(call_counter) + "_output_audio")) #Generate a audio file for each video segment based on the voice over for that narrative
                        narrative += vision_analysis1
                        vision_analysis1 = "" #reset temp variable
                        call_counter += 1
                    frame_counter += 1
                #print(narrative)
                audio_file_name = text_to_audio(narrative)
        output_filename = merge_audio_video(uploaded_file.name,"output_audio.wav", "final_output.mp4")
        # Display the result
        st.video(output_filename)

    # Clean up the temporary files
    #os.unlink(video_filename)
    #os.unlink(audio_filename)
    #os.unlink(final_video_filename)



if __name__ == '__main__':
    main()