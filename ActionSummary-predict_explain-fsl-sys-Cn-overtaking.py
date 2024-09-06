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
import video_utilities as vu
from jinja2 import Environment, FileSystemLoader

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
# azure_vision_key=os.environ["AZURE_VISION_KEY"]

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
def AnalyzeVideo(vp,fi,fpi,face_rec=False):  #fpi is frames per interval, fi is frame interval
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
    def send_post_request(resource_name, deployment_name, api_key, data):
        url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2024-06-01"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }

        print(f"resource_name: {resource_name}")
        print(f"Sending POST request to {url}")
        print(f"Headers: {headers}")
        # print(f"Data: {json.dumps(data)}")
        # print(f"api_key: {api_key}")
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
            
        # # Convert final_arr to JSON string and add it as context. final_arr might mislead the model, so commenting it out
        # if final_arr:
        #     # Only keep the last 2 results
        #     recent_context = final_arr[-2:]
        #     previous_responses = json.dumps(recent_context, ensure_ascii=False, indent=2)
        #     cont.insert(0, {
        #         "type": "text",
        #         "text": f"Previous analysis context: {previous_responses}"
        #     })
        #     # print("cont:", cont[0:2])

        #add few-shot learning metadata
        def image_to_base64(image_path):
            image = cv2.imread(image_path)
            image = vu.resize_down_to_256_max_dim(image) # resize to small image
            new_path = os.path.basename(image_path).split('.')[0] + '_resized.jpg'
            cv2.imwrite(new_path, image, [int(cv2.IMWRITE_JPEG_QUALITY),30]) #70
            image = cv2.imread(new_path)
            _, buffer = cv2.imencode('.jpg', image)

            base64_str = base64.b64encode(buffer).decode('utf-8')
            return base64_str
        
        example_images_relativeposition = {
            "redtruck-32s.png": image_to_base64("./fsl/redtruck-32s.png"),
            "redtruck-33s.png": image_to_base64("./fsl/redtruck-33s.png"),
        }
        example_images_lowerbar = {
            "lowerbar_1.png": image_to_base64("./fsl/lowerbar_1.jpg"),
            "lowerbar_3.png": image_to_base64("./fsl/lowerbar_3.jpg"),
            "lowerbar_5.png": image_to_base64("./fsl/lowerbar_5.jpg"),
            "lowerbar_7.png": image_to_base64("./fsl/lowerbar_7.jpg"),
        }
        example_images_raisebar = {
            "lowerbar_7.png": image_to_base64("./fsl/lowerbar_7.jpg"),
            "lowerbar_8s.png": image_to_base64("./fsl/lowerbar_8s.jpg"),
            "lowerbar_9s.png": image_to_base64("./fsl/lowerbar_9s.jpg"),
            "lowerbar_10s.png": image_to_base64("./fsl/lowerbar_10s.jpg"),
        }
        # 加载image label JSON文件
        assistant_response_relativeposition = """
            {
                "Start_Timestamp": "s",
                "sentiment": "Negative",
                "End_Timestamp": "s",
                "scene_theme": "Dramatic",
                "characters": "驾驶红色卡车的司机",
                "summary": "在这段视频中，车辆行驶在一条有高架桥的道路上。前方有几辆车在排队等候。突然，左侧一辆红色卡车突然从靠近自车左侧的黑色卡车左后方窜出，撞上了高架桥的桥墩。",
                "actions": "自车保持低速行驶，可能是因为前方有车辆排队等候，司机需要小心驾驶，以避免与失控车辆发生碰撞。自车在最后一帧静止，距离前车5米.",
                "key_objects": "1) 左侧：一辆红色卡车，急速左转并撞上了高架桥的桥墩，距离较近，大约5米，。",
                "key_actions": "撞桥墩",
                "next_action": "由于左侧有一辆红色卡车失控并撞上了高架桥的桥墩，建议司机减速并向右移动，以确保安全。下一步行动:速度控制: 减速, 方向控制: 转动方向盘绕行, 车道控制: 向右移动"
            }
           """ 
        assistant_response_lowerbar = """
            {
            "Start_Timestamp": "s",
            "sentiment": "Negative",
            "End_Timestamp": "s",
            "scene_theme": "Dramatic",
            "characters": "闸杆",
            "summary": "停车场的闸杆一开始在上面大部分打开状态,从上向下缓慢落下直到关闭,但没有完全关闭。",
            "actions": "停车场的闸杆处于\"落下过程中\"状态\", 到达\"接近关闭\"状态",
            "key_objects": "闸杆",
            "key_actions": "落杆",
            "next_action": "由于闸杆处于从高处落下至快要关闭，建议司机不要通过闸杆。下一步行动:速度控制: 等待, 方向控制: 保持方向, 车道控制: 保持在当前车道"      
        }
        """
        assistant_response_raisebar = """
            {
            "Start_Timestamp": "s",
            "sentiment": "Negative",
            "End_Timestamp": "s",
            "scene_theme": "Dramatic",
            "characters": "闸杆",
            "summary": "停车场的闸杆从下到上逐渐抬起,一开始是关闭,后面逐渐打开,但没有完全打开。",
            "actions": "停车场的闸杆正从低处逐渐抬起打开，处于\"抬起过程中\"状态， 尚未达到完全开启。",
            "key_objects": "闸杆",
            "key_actions": "抬杆",
            "next_action": "由于闸杆处于从低处逐渐抬起，建议司机不要通过闸杆。下一步行动:速度控制: 等待, 方向控制: 保持方向, 车道控制: 保持在当前车道"      
        }
        """
        # setting template format with jinja2
        env = Environment(loader=FileSystemLoader('./fsl'))
        template = env.get_template('fewshot_userpmt.jinja2')
        # render the template
        fsl_payload = template.render(example_images=example_images_relativeposition)
        lowerbar_fsl_payload = template.render(example_images=example_images_lowerbar)
        raisebar_fsl_payload = template.render(example_images=example_images_raisebar)

        #extraction template

        
        
        if(vision_api_type=="Azure"):
            print("sending request to gpt-4o")
            payload2 = {

                "messages": [
                    {
                    "role": "system",
                    "content": f"""You are VideoAnalyzerGPT analyzing a series of SEQUENCIAL images taken from a video, where each image represents a consecutive moment in time.Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle. Pay special attention to any signs of deceleration or closing distance between the car in front and the observer vehicle(self car). Describe any changes in the car's speed, distance from the observer vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take. Consider the current distance between the front vehicle and self vehicle, the speed of the car in front, and any changes in these factors. If the car ahead is decelerating and the distance is closing rapidly, suggest whether braking is necessary to avoid a collision. Examine the sequential images for visual cues that indicate the car in front is decelerating, such as the appearance of brake lights or a reduction in the gap between the vehicles. If the self car is overtaking the front car or an adjacent car, also summarize in the summary field. Don't ignore the self car's overtaking due to no danger happening, append in the summary and key_actions. Consider how these cues change from one frame to the next, and describe the need for the observer vehicle to take action, such as braking, based on these changes.

                    Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
                    as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
                    You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
                    frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until we have a full action summary of the portion of the video you are considering,
                    that includes all actions taken by characters in the video as extracted from the frames. 
                    Direction - Please identify the objects in the image based on their position in the image itself. Do not assume your own position within the image. Treat the left side of the image as 'left' and the right side of the image as 'right'. Assume the viewpoint is standing from at the bottom center of the image. Describe whether the objects are on the left or right side of this central point, left is left, and right is right. For example, if there is a car on the left side of the image, state that the car is on the left.
                    Self and other vehicle movement determination  - please pay attention to determine whether it is the self-driving car or the other car that is moving:
                        1. Observe the relative position changes of background objects (such as road signs, buildings, etc.).
                        2. Check the relative position changes of vehicles in consecutive frames.
                    Example: If the background is moving, the self-driving car is changing lanes. The relative movement of trees in the background indicates that the self-driving car is changing lanes.

                    **Task 1: Identify and Predict potential very near future time 自车超车,etc Behavior**
                    Self vehicle overtaking(自车超车): The definition of a vehicle overtaking refers to the maneuver performed by the driver in which the vehicle accelerates to pass a slower-moving vehicle ahead from one side and then either returns to the original lane in front of that vehicle or continues driving in the other lane. Overtaking can also involve quickly passing a vehicle in an adjacent lane. In this case, the vehicle might not necessarily return to its original lane but may remain in the lane after overtaking. This action is typically done when the road ahead is safe and there is sufficient space to increase driving efficiency or avoid traffic congestion.Put "自车超车" in the "key_actions" field if detected, no matter if it is a dangerous action! If "overtaking" is in summary, you must put it in "key_actions"!!! Methods to identify a self car overtaking can be referenced by observing the following positional changes: a)Lane Change: Overtaking is typically accompanied by the vehicle changing lanes from one to an adjacent lane. On a dual-lane road, the vehicle will shift from the slow lane to the fast lane to overtake; on a single-lane, two-way road, the vehicle will briefly enter the oncoming lane to pass. b)Acceleration: During overtaking, the vehicle's speed usually increases to quickly pass the vehicle in front. c)Change in Distance to the Vehicle Ahead: A key indicator of overtaking is the noticeable reduction in the distance between the vehicle and the one ahead, followed by a swift pass and an increase in the distance from the overtaken vehicle.自车超车可能不是危险动作,只是为了提高行车效率或避免交通拥堵,但是无论是否危险,只要在summary中出现了"超车,超过前面的车"这种类似的词,你就必须在"key_actions"中报告这个动作!!!

                    
                    Provide detailed description of the people's behavior and potential dangerous actions that could lead to collisions. Describe how you think the individual could crash into the car, and explain your deduction process. Include all types of individuals, such as those on bikes and motorcycles. 
                    Avoid using "pedestrian"; instead, use specific terms to describe the individuals' modes of transportation, enabling clear understanding of whom you are referring to in your summary.
                    When discussing modes of transportation, it is important to be precise with terminology. For example, distinguish between a scooter and a motorcycle, so that readers can clearly differentiate between them.
                    Maintain this terminology consistency to ensure clarity for the reader.
                    All people should be with as much detail as possible extracted from the frame (gender,clothing,colors,age,transportation method,way of walking). Be incredibly detailed. Output in the "summary" field of the JSON format template.
                    
                    **Task 2: Explain Current Driving Actions**
                    Analyze the current video frames to extract actions. Describe not only the actions themselves but also provide detailed reasoning for why the vehicle is taking these actions, such as changes in speed and direction. Focus solely on the reasoning for the vehicle's actions, excluding any descriptions of pedestrian behavior. Explain why the driver is driving at a certain speed, making turns, or stopping. Your goal is to provide a comprehensive understanding of the vehicle's behavior based on the visual data. Output in the "actions" field of the JSON format template.

                    **Task 3: Predict Next Driving Action**
                    Understand the current road conditions, the driving behavior, and to predict the next driving action. Analyze the video and audio to provide a comprehensive summary of the road conditions, including weather, traffic density, road obstacles, and traffic light if visible. Predict the next driving action based on two dimensions, one is driving speed control, such as accelerating, braking, turning, or stopping, the other one is to predict the next lane control, such as change to left lane, change to right lane, keep left in this lane, keep right in this lane, keep straight. Your summary should help understand not only what is happening at the moment but also what is likely to happen next with logical reasoning. The principle is safety first, so the prediction action should prioritize the driver's safety and secondly the pedestrians' safety. Be incredibly detailed. Output in the "next_action" field of the JSON format template.

                    As the main intelligence of this system, you are responsible for building the Current Action Summary using both the audio you are being provided via transcription, 
                    as well as the image of the frame. 
                    Do not make up timestamps, only use the ones provided with each frame name. 

                    Your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON.
                    你现在是一名中文助手。无论我问什么问题，你都必须只用中文回答。请不要使用任何其他语言。You must always and only answer totally in **Chinese** language!!! I can only read Chinese language. Ensure all parts of the JSON output, including **summaries**, **actions**, and **next_action**, **MUST BE IN CHINESE** If you answer ANY word in English, you are fired immediately! Translate English to Chinese if there is English in "next_action" field.
                    
                    Example:
                        "Start_Timestamp": "s",
                        "sentiment": "Negative",
                        "End_Timestamp": "s",
                        "scene_theme": "Dramatic",
                        "characters": "自车司机",
                        "summary": "在这段视频中，自车超车超过了前方车辆,并躲过了摔倒的白色摩托车。",
                        "actions": "自车从左侧超过了前方的黄色拉土车.",
                        "key_objects": "1) 正前方：一辆黄色拉土车，距离较近，大约10米.",
                        "key_actions": "自车超车",
                        "next_action": "由于前方有一辆黄色拉土车行驶在路上，建议司机超车后保持车速。下一步行动:速度控制: 维持现在速度, 方向控制: 不变, 车道控制: 不变"
                        
                        
                    # Few-shot learning metadata
                    Below are time series example images and their corresponding analysis to help you understand how to analyze and label the images:
                    {fsl_payload} -> {assistant_response_relativeposition}
                            
                    Use these examples to understand how to analyze and analyze the new images. Now generate a similar JSON response for the following video analysis:
                    """
                    },
                    {"role": "user", "content": cont}
                ],
                "max_tokens": 2000,
                "seed": 42,
                "temperature": 0
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
                    your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON.
                    Answer in Chinese language."""
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
    print(f"Frames per second: {fps}")

    # Process video
    current_frame = 0
    current_second = 0
    current_summary=""

    # Load video audio
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration  # Duration of the video in seconds
    print(f"Video duration: {video_duration} seconds")

    # Process video
    current_frame = 0  # Current frame initialized to 0
    current_second = 0  # Current second initialized to 0
    current_summary=""
    packet=[] # Initialize packet to hold frames and audio
    current_interval_start_second = 0
    capture_interval_in_frames = int(fps * frame_interval / frames_per_interval)  # Interval in frames to capture the image
    # capture_interval_in_frames = int(fps * frame_interval / frames_per_interval)  # Interval in frames to capture the image
    capture_interval_in_seconds = frame_interval/frames_per_interval  # 每0.2秒捕捉一帧
    print(f"Capture interval in seconds: {capture_interval_in_seconds}")
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
                print("frame_name: ",frame_name, ", current_frame: ", current_frame, ", current_second: ", current_second)
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
            # time.sleep(5)
            try:
                vision_analysis = vision_response["choices"][0]["message"]["content"]
                if not vision_analysis:
                    print("No video analysis data")
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
                vault = vault.replace("\\\\n", "\\n").replace("\\n", "\n")  # 将转义的 \n 替换为实际换行符

                if vault.startswith('json'):
                # Remove the first occurrence of 'json' from the response text
                    vault = vault[4:]
                    #print(vault)
                else:
                    vault = vault
                #print(vision_analysis.replace("'",""))
                data=json.loads(vault, strict=False) #If strict is False (True is the default), then control characters will be allowed inside strings.Control characters in this context are those with character codes in the 0-31 range, including '\t' (tab), '\n', '\r' and '\0'.
                #print(data)
                final_arr.append(data)
                if not data:
                    print("No data")
                # for key, value in data:
                #     final_arr.append(item)
                #     ##socket.emit('actionsummary', {'data': item}, namespace='/test')
                #     print(f"Key: {key}, Value: {value}")

                with open('actionSummary.json', 'w', encoding='utf-8') as f:
                # Write the data to the file in JSON format
                    json.dump(final_arr, f, indent=4, ensure_ascii=False) #ensure_ascii=False to write in Chinese
                    print(f"Data written to file: {final_arr}") # 调试信息

            except:
                miss_arr.append(vision_analysis)
                print("missed")

            spinner.stop()
            spinner = Spinner("Capturing Video and Audio...")
            spinner.start()

            packet.clear()  # Clear packet after analysis
            current_interval_start_second += frame_interval  # Move to the next set of frames

        if current_second > video_duration:
            print("Current second is: ", current_second), "Video duration is: ", video_duration, "Exiting loop"
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
        json.dump(final_arr, f, indent=4, ensure_ascii=False)
        
    with open('transcript.txt', 'w') as f:
    # Write the data to the file in JSON format
        f.write(global_transcript)
    return final_arr

    #print("\n\n\n"+totalData)

#AnalyzeVideo("./medal.mp4",60,10)
# AnalyzeVideo("Nuke.mp4",60,10,False)
# AnalyzeVideo("复杂场景.mov",1,10) # 10 frames per interval, 1 second interval. fpi=10 is good for accurate analysis
# AnalyzeVideo("test_video/三轮车.mp4",4,10)
# AnalyzeVideo("test_video/鬼探头2.mov",2,10)
# AnalyzeVideo("test_video/0_cutin_aeb.mp4",2,10)
# AnalyzeVideo("时序事件类/抬杆/负样本3.mp4",15,8)
# AnalyzeVideo("trim落下.mp4",9,10)
# AnalyzeVideo("时序事件类/横穿马路/负样本5.mp4",2,10)
AnalyzeVideo("时序事件类/自车超车/正样本4.mp4",6,10)
# AnalyzeVideo("时序事件类/自车超车/负样本5.mp4",12,10)

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