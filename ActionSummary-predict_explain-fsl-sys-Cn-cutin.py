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
                        "detail":"low"
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
        json_form=str([json.dumps({"Start_Timestamp":"4s (the second number in the path name of the frame file, eg, 0.2s)",\
                                "sentiment":"Positive, Negative, or Neutral (reflect the urgency of the situation to accurately convey the level of imminent danger in the scene)",\
                                "End_Timestamp":"5s (the second number in the path name of the frame file, eg, 1s)",\
                                "scene_theme":"Dramatic",\
                                "characters":"Man in hat, woman in jacket",\
                                "summary":"Summary of events occurring around these timestamps including actions. Use both transcript and frames to create a full picture, be detailed and attentive, be serious and straightforward. Ensure to distinguish whether the action is performed by the car itself or other cars. Self vehicle overtaking the car ahead is an important action of self car, don't omit due to observing the other cars or surroundings, make sure keep an eye on the self car's overtaking other cars.自车超车一旦发生,就要在summary里上报\
                                           Please carefully analyze the key objects and their positions in the image, considering their relative relationships and interactions with the surrounding environment.",\
                                "actions":"Extract key actions via both video frames analysis and summary above, also use transcript. If a dangerous accident is already mentioned in the summary above, you must include the IMPORTANT event in the actions as well. Ensure to distinguish whether the action is performed by the car itself or other cars, don't confuse the subjects of events occuring in the video and audio. The events described in the audio, such as collision, could involve our vehicle or another vehicle.It must be determined in conjunction with the video. If there is any potential for misunderstanding, provide an explanation to clarify who is performing the action. Additionally, you should not only describe the actions themselves, but also provide detailed reasoning for why the vehicle is taking the current actions, such as why the driver is driving in low speed or changes in speed and direction. Report self vehicle's speed.Focus soly on the reasoning for the vehicle's action, excluding any pedestrian's behavior.",\
                                "key_objects": "Key objects are those that could **immediately affect the vehicle’s path or safety**, such as **moving vehicles**, **pedestrians stepping into the road**, or **roadblocks** that necessitate a sudden maneuver. **Focus solely on objects that pose a direct threat** to the safety of the car. **Exclude any objects** that do not present a clear and present danger, such as trees or mounds beside the road, as these do not pose a direct threat. For example, **do not include static natural objects** like trees, grass, or roads unless they are **directly involved in a potential collision scenario** (e.g., a tree falling towards the road). **Do not include objects that are static and pose no immediate threat**, such as trees or bushes beside the road or roads. \
                                    1) **Describe its relative orientation** to the car itself, e.g., left, right, middle position of the car. \
                                    2) **Estimate the distance** between the object and the car itself in meters, accurate to one decimal place at most. **Always include a specific value**, regardless of how close you perceive the object to be. \
                                    3) **Describe their next behavior** direction, such as whether they are likely to accelerate or slow down. \
                                    4) **DO NOT describe any unimportant objects** that are irrelevant to the car itself. \
                                    The **key objects** include not only people but also dangerous barriers such as **roadblocks** and **water puddles**. **Focus on detailing all relevant and dangerous objects**, with as much detail as possible extracted from the frame (**gender**, **clothing**, **colors**, **age**). **Be incredibly detailed**. **Include colors along with descriptions**.",\
                                "key_actions": "List dangerous actions if detected from both 'summary' and 'actions'. 如果上面的'summary'里发现了关键行为，如突然停车、突然减速、超车/超过某车，就要列在key_actions里面，不要不报，否则就算一次失败的任务!!!! IF any key behavior is mentioned in the summary or actions, such as sudden braking, sudden deceleration, or vehicle stopping of both front/near-by cars and self car, or overtaking, you must report it in the key_actions,otherwise it is counted as a failed request. Do not omit such actions under any circumstances. Include the following actions but not limited to: Ghosting probing(鬼探头); cut in(加塞,pay special attention to any vehicle suddenly cutting in from the side lane in front of the car. Analyze the changes in vehicle positions in each frame of the video and determine if any vehicle suddenly cuts in front of the car.);前车急刹车;Scraping(剐蹭); lane change(变道);紧急变道 (Emergency Lane Change);急变道 (Abrupt Lane Change); 失控(Loss of Control);overtaking(超车); Dangerous Overtaking(危险超车);Rapid Acceleration (急加速);sudden appearance(突然出现); passing through(穿过);斜穿(Diagonal Crossing);rapidly approaching(快速接近);Tailgating(跟车太近); Side-by-Side Close Proximit(侧面贴近);Wrong-Way Driving (逆行);Running Red Light (闯红灯)；撞车 (Collision.Estimate the distance between other car and self car, if too close then infer collision happening possibility);撞击 (Impact);Reversing (倒车);偏离车道 (Lane Departure),转弯(make a turn),自车掉头(U-turn of the self vehicle),自车超车(Self vehicle overtaking), etc. \
                                    Pay special attention to any sudden or unexpected changes in the behavior of vehicles in the vicinity of the car, particularly those that may indicate a 'cut in' (加塞) or sudden deceleration. 'Cut in' refers to any vehicle that suddenly and forcefully merges into the lane in front of the car, potentially creating a hazard. Analyze the changes in vehicle positions in each frame of the video, especially if a vehicle moves into the lane directly in front of the car and decelerates. This includes recognizing situations where a vehicle that was previously in another lane suddenly moves in front of the car and reduces its speed significantly. Ensure to distinguish between normal lane changes and 'cut in' by assessing the urgency and impact on the car's safety. Analyze whether there is sufficient space for the vehicle to merge into your lane, and if so, classify this as a 'cut in' (加塞) in the key_actions. Example: If a vehicle suddenly moves from a side lane into the lane directly in front of the car and begins to decelerate rapidly, this should be identified as a 'cut in' (加塞) in the key actions. The key actions field should include 'cut in' when this behavior is detected, particularly if the distance between the two vehicles decreases sharply and the car needs to take evasive action to maintain safety. \
                                    Do not confuse sudden pedestrian appearance (ghost probing) with cutting in; the difference is that ghost probing(鬼探头) involves pedestrians or non-motorized vehicles suddenly appearing, while cutting in involves a vehicle forcibly merging into traffic. \
                                    Key actions should include what you see other cars' action and the car itself's key action based on your reasoning as well. Only fill in the field when detecting dangers that might impact the car itself driving, do not misuse!\
                                    If multiple key actions occur, list all of them and rank them according to the level of risk they pose to the vehicle.\
                                    Before finalizing the key_actions, review the summary and actions to ensure that all critical behaviors mentioned there, such as sudden deceleration, stopping, or any other dangerous actions, are reflected in the key_actions. This step is mandatory to maintain consistency between summary and key_actions.Example: If the summary states that 'the silver car in front gradually slowed down and stopped then the key_actions must include sudden deceleration' or 'stopping' as key_actions. Ensure these key behaviors are consistently reflected in both summary and key_actions. 'key_actions' like 'sudden appearance' etc must be identified regardless of the sentiment or scene_theme context, ensure that this action is emphasized, you should prioritize identifying key_actions listed above. IF no above key_actions in the list are identified or identified in \"Summary\" field, just leave the field as '无危险行为'.", \
                                "next_action":"Predict next driving action (e.g., accelerate, brake, turn left, turn right, stop), considering key objects impacting driving action, use both transcript and frames to create a full picture, reason for the predicted next driving action. Note: \
                                    1) a.Confirm the positions of both the other vehicle and our vehicle at the end of the time interval. b)Consider the movement direction and speed of both vehicles during the interval. Based on these factors, determine the next action. \
                                        For example: The vehicle ahead moved from the right to the left within 5 seconds, and our vehicle is preparing to turn left, so the instruction should be to brake early to avoid a collision. \
                                    2) Ensure that the next_action predictions and descriptions for the vehicle's future behavior are consistent. For instance, if a vehicle suddenly appears on the left, then the next action should be to turn right to avoid collision. After the above prediction, summarize the next_action with following format but totally in Chinese and divided into three main categories, for example: \
                                    (Please provide the JSON response without escaped newline characters (\\n), this will cause serious error.) \
                                    Translate English to Chinese if there is any English words in summarization. \
                                  - 速度控制：加速、减速、快速减速、慢速直行、匀速直行、停车、等待、倒车  \
                                  - 方向控制：左转、右转、掉头、转动方向盘绕行、刹车(如发生了碰撞，1)若周围情况允许，则刹车并检查碰撞的严重程度如何;2)如在高速上，如果车辆仍然可以行驶，慢慢驶向路肩或紧急停车带，尽量将车辆停在安全位置，远离行车道;3)如果车辆无法移动，保持在当前位置。但具体情况仍要具体分析，视现场和危险程度而决定是否刹车) \
                                  - 车道控制：向左变道、向右变道、稍微向左移动、稍微向右移动、回到正常车道 \
                                Note:  \
                                - Speed Control: \
                                Acceleration: When merging onto highways, overtaking another vehicle, or speeding up to match traffic flow.  \
                                Deceleration: When approaching a red light, stop sign, or slow-moving traffic. \
                                Rapid Deceleration: In emergency situations to avoid collisions or sudden obstacles. \
                                Slow Steady Driving: In heavy traffic, residential areas, or school zones.  \
                                Consistent Speed Driving: On highways or roads with steady traffic flow. \
                                Stopping: At red lights, stop signs, pedestrian crossings, or in case of an emergency.  \
                                Waiting: In traffic jams, at intersections, or while yielding.   \
                                Reversing: When parking, maneuvering in tight spaces, or exiting a driveway. \
                                - Direction Control: \
                                Left Turn: At intersections, driveways, or parking lots. \
                                Right Turn: At intersections, driveways, or parking lots. \
                                U-Turn: When you need to change direction completely, usually at intersections or designated U-turn areas.  \
                                Steering to Circumvent: When avoiding obstacles, roadwork, or debris on the road.  \
                                - Lane Control:   \
                                Changing to the Left Lane: When preparing for a left turn, overtaking a slower vehicle, or moving into a faster lane.  \
                                Changing to the Right Lane: When preparing for a right turn, exiting a highway, or moving into a slower lane.   \
                                Slight Left Shift: To avoid small obstacles, adjust position in lane, or respond to road conditions.   \
                                Slight Right Shift: To avoid small obstacles, adjust position in lane, or respond to road conditions.    \
                                Totally in Chinese language.Translate English to Chinese if there is any English words in summarization."})])
        
        
        if(vision_api_type=="Azure"):
            print("sending request to gpt-4o")
            payload2 = {

                "messages": [
                    {
                    "role": "system",
                    "content": f"""You are VideoAnalyzerGPT analyzing a series of SEQUENCIAL images taken from a video, where each image represents a consecutive moment in time.Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle. Pay special attention to any signs of deceleration or closing distance between the car in front and the observer vehicle(self car). Describe any changes in the car's speed, distance from the observer vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take. Consider the current distance between the front vehicle and self vehicle, the speed of the car in front, and any changes in these factors. If the car ahead is decelerating and the distance is closing rapidly, suggest whether braking is necessary to avoid a collision. Examine the sequential images for visual cues that indicate the car in front is decelerating, such as the appearance of brake lights or a reduction in the gap between the vehicles. If the self car is overtaking the front car or an adjacent car, also summarize in the summary field. Don't ignore the self car's overtaking due to no danger happening, append in the summary and key_actions. Consider how these cues change from one frame to the next, and describe the need for the observer vehicle to take action, such as braking, based on these changes.
                    不要轻易使用”自车保持了与前车的安全距离“，前车与自车的安全距离判定要根据“两秒规则”（Two-Second Rule）: 不要判断图片中的前车和其他车之间的距离，只要判断自己车辆与前车之间的距离。
                    适用情况: 良好天气条件下的普通道路。
                    规则: 在车辆行驶时，选择前方车辆经过的某一个固定点，如路标或树木。当前方车辆经过该点时，开始计时，确保自己车辆经过该点时，至少已经过了两秒钟。这相当于在大多数情况下以两秒的时间作为安全距离。
                    解释: 这种方法确保驾驶员有足够的时间和距离来应对前方车辆的突然减速或紧急情况。
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

                    
                    **Task 1: Identify and Predict potential very near future time "Ghosting(专业术语：鬼探头)" 、Cut-in(加塞) .,etc Behavior**
                    1)"Ghosting" behavior refers to a person suddenly darting out from either left or right side of the car itself and also from BEHIND an object that blocks the driver's view, such as a parked car, a tree, or a billboard, directly into the driver's path. 1)Note that people coming straight towards the car from front are not considered as 鬼探头, only those coming from visual blind spot of the car itself can be considered as 鬼探头. 2)Note that cut-in加塞 is different from 鬼探头,
                    Cutting In:
                        Definition: When a vehicle deliberately forces its way in front of another vehicle or into a traffic lane.开车加塞是指在行驶过程中，某辆车强行插入其他车辆的行驶路线，这种情况下一般是指距离非常近，从而影响其他车辆的正常行驶，甚至导致紧急刹车。如果行驶速度正常并且前后车的间隙明显不够，还强行加塞就属于恶意加塞。
                        Risk: Can lead to sudden braking and rear-end collisions.
                             判断自车与前车的距离：看前挡风玻璃下沿，看到前车后保险杠的上沿，车距为1米；看到前车后保险杠的下沿，车距为2米，看到前车后轮胎的下沿，车距为3米。停车时，前方停止线和左前门角5CM处对正，刚好不越线。
                        安全车距：一般情况下，车辆之间的安全车距可以用“车速/2”来估算。例如，车速为100公里/小时时，安全车距应当保持至少50米。但在加塞时，插入的距离往往会小于这个安全距离。所以你先估计目前的车速是多少，然后判断前车与自己的距离是否大于“车速/2”。如果小于这个值，就说明前车与自己的距离不够，这时候就要注意观察前车的动向，以防发生加塞，并报告这个加塞行为在key_actions中。     
                    Note：正常变道跟加塞的区别
                        1、正常变道是司机在交通路况正常的情况下变更车道，并且不会影响后方车辆的行驶。
                        2、而加塞则是司机不正常或者恶意的变道，这样对后方车辆会造成不良影响，可能还会导致后方车辆的突然减速从而引起交通的堵塞，出现塞车的状况。         
                    Sudden Pedestrian Appearance (Ghost Probing):
                        Definition: When a pedestrian, cyclist, or another vehicle suddenly emerges from behind an obstacle into the driver's path.
                        Characteristics:
                            Unexpected event, usually involving non-drivers.
                            Almost no reaction time for the driver.
                            Common in residential areas, near schools, or at intersections.
                        Risk: High chance of collision due to the suddenness of the event. 
                    This behavior usually occurs when individuals, either not paying attention to the traffic conditions or in a hurry, 
                    suddenly obstruct the driver's view, creating an emergency situation with very little reaction time. This can easily lead to traffic accidents.
                    2) Raising the parking barrier(抬杆) refer to the action of lifting a physical barrier, often found at the entrance or exit of parking lots, gated communities, or secured areas. This barrier controls vehicle access, and raising it allows vehicles to pass through. The barrier is typically operated manually by a security guard or automatically by a sensor or remote control. This process is essential for managing the flow of traffic and ensuring that only authorized vehicles can enter or exit the controlled area.
                    There are the distinctions between the different states of the parking barrier, which are as follows, and you should put the exact state of the parking barrier in the "actions" field, and put "抬杆"/"落杆" in the "key_actions" field as a category,if both 抬杆and落杆happen in the same video,you should report both of them, eg,"抬杆, 落杆":
                    一直抬起(continuously raised):
                    the barrier remains fully lifted at all times. In this position, the barrier does not obstruct the passage of vehicles, typically indicating that the area is open for free access or that the control system is temporarily out of service and not functioning properly.
                    抬起过程中 (In the Process of Raising barrier):
                    The parking barrier is in the act of being lifted from a lower position toward the fully raised position. Vehicles cannot yet pass through as the barrier is not fully open.
                    落下过程中 (In the Process of Lowering barrier):
                    The parking barrier is in the act of being lowered from the fully raised position toward the closed position. Vehicles cannot pass through as the barrier is in motion and partially obstructs the path.
                    一直关闭 (continuously Closed barrier)--如果闸杆一直没动，对应的动作和状态不是"落杆",而是"一直关闭",key_actions应该是空的:
                    The parking barrier is completely lowered and positioned vertically or close to vertical, fully blocking the passage of vehicles. This prevents unauthorized entry or exit.
                    Note that: 1.The normal sequence of raising and lowering the barrier is as follows: typically, the barrier remains in a continuously closed state. When a vehicle approaches, the barrier raises, entering the "in the process of raising" state. Once fully raised, it stays in the "continuously raised" state, allowing the vehicle to pass. Then, it gradually lowers, entering the "in the process of lowering" state, and finally returns to the "continuously closed" state. Detect the movement direction of the barrier in consecutive frames with progressively advancing timestamps. If the position of the barrier between two frames is gradually rising from a lower position to a higher position, it can be identified as the process of RAISING the barrier抬杆. If the position is gradually lowering from a higher position to a lower position, it can be identified as the process of LOWERING the barrier落杆.如果开始时闸杆是关闭状态且最终是打开状态，则这个过程应该是抬杆；反之则为落杆，不要混淆!!!!
                         2.Don't confuse the garage's "rolling shutter door" with the "barrier gate." The rolling shutter door rolls up and down, while the barrier gate moves up and down; they are different. The rolling shutter door, when closed, presents a complete facade and is typically used to secure entrances like garages and shops.The key actions for the rolling shutter door should be called "door up" and "door down," not "bar up" and "bar down"!!!!!不要混淆车库的"卷闸门"和"抬杆"，卷闸门是上下卷动的，抬杆是上下移动的，两者不同。卷闸门在关闭时呈现为一个完整的门面，通常用于封闭车库、商店等出入口。卷闸门的key_actions应该叫"抬门","落门",卷闸门升降不要叫"抬杆","落杆"!!!!!
                        3.If the barrier is in a continuously closed state, the key_actions field should be empty,no matter how the angle is changed to view it, and there is no action to report.The barrier arm will not necessarily lift just because it remains closed for a long time. Do not make incorrect assumptions.如果闸杆一直是水平也就是关闭状态，不论从哪个角度观察key_actions都应该是空的，没有动作需要报告,不要识别成"抬杆"，不要错误推测为闸杆一直关闭后就要抬起。

                    3)U-turn of the self vehicle(自车掉头): The self vehicle makes a U-turn, reversing its direction by 180 degrees. This action is typically performed on wide roads or at intersections with a large turning radius. The U-turn is executed by turning the steering wheel to the left or right, depending on the direction of the desired turn, and completing a full rotation to face the opposite direction. This maneuver is used when the driver needs to change course or return to the starting point. Put "自车掉头" in the "key_actions" field if detected. Methods to identify a U-turn can be referenced by observing the following positional changes: a)Change in Vehicle Direction:Observe whether the vehicle's driving direction has made a significant 180-degree turn. A U-turn typically means the vehicle is turning completely in the opposite direction from its original path. This is the most direct sign of a U-turn.
                    b)Steering Angle:Pay attention to the vehicle's steering during the turn. If the vehicle is making a large turn (usually close to 180 degrees), it is likely performing a U-turn.
                    c) Lane Position: A U-turn usually occurs in the middle of the road, where the vehicle crosses from one lane to the opposite lane. Therefore, observe if the vehicle is moving from its original lane to the opposite lane.
                    d)Change in the vehicle’s position relative to road signs or landmarks: A U-turn will cause a noticeable change in the vehicle’s relative position, such as moving from one side of a landmark to the other, or shifting from one lane to the opposite lane. These changes indicate a significant change in the vehicle’s direction.You need to identify that multiple small turns combined result in a total angle of 180 degrees, which is a characteristic of a vehicle's U-turn.车辆在掉头时可能不会立即进行一个明显的180度转弯，而是通过多个小幅度的方向调整逐渐完成掉头。这使得掉头的过程不易被迅速察觉，你要总结出多个小转弯合起来的角度达到180度，就是自车掉头的特征,这不是仅仅对视频画面的总结，而应该更进一步对打满方向盘的画面进行车头方向变化、方向盘转动角度、车窗外背景变化、车道变化等多种因子分析是否发生了掉头。

                    4)Self vehicle overtaking(自车超车): The definition of a vehicle overtaking refers to the maneuver performed by the driver in which the vehicle accelerates to pass a slower-moving vehicle ahead from one side and then either returns to the original lane in front of that vehicle or continues driving in the other lane. Overtaking can also involve quickly passing a vehicle in an adjacent lane. In this case, the vehicle might not necessarily return to its original lane but may remain in the lane after overtaking. This action is typically done when the road ahead is safe and there is sufficient space to increase driving efficiency or avoid traffic congestion.Put "自车超车" in the "key_actions" field if detected, no matter if it is a dangerous action! If "overtaking" is in summary, you must put it in "key_actions"!!! Methods to identify a self car overtaking can be referenced by observing the following positional changes: a)Lane Change: Overtaking is typically accompanied by the vehicle changing lanes from one to an adjacent lane. On a dual-lane road, the vehicle will shift from the slow lane to the fast lane to overtake; on a single-lane, two-way road, the vehicle will briefly enter the oncoming lane to pass. b)Acceleration: During overtaking, the vehicle's speed usually increases to quickly pass the vehicle in front. c)Change in Distance to the Vehicle Ahead: A key indicator of overtaking is the noticeable reduction in the distance between the vehicle and the one ahead, followed by a swift pass and an increase in the distance from the overtaken vehicle.自车超车可能不是危险动作,只是为了提高行车效率或避免交通拥堵,但是无论是否危险,只要在summary中出现了"超车,超过前面的车"这种类似的词,你就必须在"key_actions"中报告这个动作!!!

                    Your angle appears to watch video frames recorded from a surveillance camera in a car. Your role should focus on detecting and predicting dangerous actions in a "Ghosting" manner
                    where pedestrians in the scene might suddenly appear in front of the current car. This could happen if the pedestrian suddenly darts out from behind an obstacle in the driver's view.
                    This behavior is extremly dangerous because it gives the driver very little time to react. 
                    Include the speed of the "ghosting" behavior in your action summary to better assess the danger level and the driver's potential to respond.
                    
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
                    as well as the image of the frame. Note: . Always and only return as your output the updated Current Action Summary in format template ```{json_form}```. 
                    Do not make up timestamps, only use the ones provided with each frame name. 

                    Your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON.
                    你现在是一名中文助手。无论我问什么问题，你都必须只用中文回答。请不要使用任何其他语言。You must always and only answer totally in **Chinese** language!!! I can only read Chinese language. Ensure all parts of the JSON output, including **summaries**, **actions**, and **next_action**, **MUST BE IN CHINESE** If you answer ANY word in English, you are fired immediately! Translate English to Chinese if there is English in "next_action" field.
                    
                    Example:
                        "Start_Timestamp": "8.4s",
                        "sentiment": "Negative",
                        "End_Timestamp": "12.0s",
                        "scene_theme": "Dramatic",
                        "characters": "驾驶白色轿车的司机",
                        "summary": "在这段视频中，车辆行驶在一条乡村小路上。前方有一辆白色轿车停在路边。驾驶员提到前方的三轮车已经让路，但仍然发生了碰撞。",
                        "actions": "自车与三轮车发生碰撞，可能是因为与右侧三轮车距离过近. 在过去的4秒中，车辆的速度逐渐减慢，最终在最后一帧时达到XX km/h，距离前车30米.",
                        "key_objects": "1) 正前方：一辆白色轿车，距离较近，大约20米，停在路边，可能会突然启动或转向.2) 右侧：一辆红色三轮车，距离约为0，已经发生剐蹭，可能会下车处理事故，也可能突然启动继续向前行驶。",
                        "key_actions": "碰撞",
                        "next_action": "由于前方有一辆白色轿车停在路边，且道路较窄，建议司机紧急刹车。下一步行动:速度控制: 减速, 方向控制: 立即刹车, 车道控制: 稍微向左移动以避免与右侧车相撞"

                        "Start_Timestamp": "30.1s",
                        "sentiment": "Negative",
                        "End_Timestamp": "31.0s",
                        "scene_theme": "Dramatic",
                        "characters": "白色轿车",
                        "summary": "在这段视频中，车辆行驶在一条隧道内。前方有一辆白色轿车突然失控，车尾扬起大量灰尘，车身向左侧偏移，似乎撞上了隧道的墙壁。前方还有几辆车在正常行驶。",
                        "actions": "车辆保持低速行驶，可能是因为前方有一辆白色轿车失控，司机需要小心驾驶，以避免与失控车辆发生碰撞。车辆的速度逐渐减慢，在最后一帧时，车辆的速度为10 km/h，距离前车x米",
                        "key_objects": "1) 从右侧到左侧快速移动：一辆白色轿车，距离较近，大约5米，失控并向左侧偏移。",
                        "key_actions": "撞墙",
                        "next_action": "由于前方左侧有一辆白色轿车失控，建议司机减速并向右移动，以确保安全。下一步行动:速度控制: 减速, 方向控制: 转动方向盘绕行, 车道控制: 向右移动"

                        "Start_Timestamp": "40.4s",
                        "sentiment": "Negative",
                        "End_Timestamp": "44.0s",
                        "scene_theme": "Dramatic",
                        "characters": "红色轿车",
                        "summary": "在这段视频中，车辆行驶在一条多车道的道路上。前方有一辆红色轿车正在尝试变道，但其变道行为不规范，导致车辆偏离车道，最终撞上了道路右侧的护栏。视频中可以听到乘客对司机的变道行为表示不满，并质疑其驾驶状态。",
                        "actions": "车辆保持低速行驶，可能是因为前方的红色轿车在不规范变道，司机需要小心驾驶，以避免与前方车辆发生碰撞。车辆的速度逐渐减慢，在最后一帧时，车辆的速度为30 km/h",
                        "key_objects": "1) 前方：一辆红色轿车，距离较近，大约10几米，正在不规范变道，可能会突然减速或转向。2) 右侧：道路护栏，距离为0，因为自车已经撞上护栏。",
                        "key actions": "碰撞",
                        "next_action": "由于前方有一辆红色轿车在不规范变道，且可能会撞上护栏，建议司机减速，以确保安全。下一步行动:速度控制: 减速, 方向控制: 向左转动并稳定方向盘, 车道控制: 向左回到正常车道并避开右侧障碍物"

                        "Start_Timestamp": "s",
                        "sentiment": "Negative",
                        "End_Timestamp": "s",
                        "scene_theme": "Dramatic",
                        "characters": "驾驶白色轿车的司机",
                        "summary": "前方有一辆白色轿车突然从右侧车道切入到自车前方，导致自车紧急刹车。",
                        "actions": "自车紧急刹车，可能是因为前方的白色轿车突然从右侧车道切入，司机需要迅速反应以避免碰撞。",
                        "key_objects": "1) 右侧：一辆白色轿车，距离较近，大约5米，突然切入到自车前方，可能会继续加速或减速。",
                        "key_actions": "加塞(为了说明key_actions必须与summary中的关键行为的关键词必须保持一致，保证譬如突然切入提取出来含义就是加塞。如果你漏掉关键词我就认为你任务失败了，千万不要漏掉！)",
                        "next_action": "由于前方有一辆白色轿车突然切入，建议司机保持警惕，减速并保持安全距离。下一步行动:速度控制: 减速, 方向控制: 稳定方向盘, 车道控制: 保持在当前车道"
                    
                        "Start_Timestamp": "s",
                        "sentiment": "Neutral",
                        "End_Timestamp": "s",
                        "scene_theme": "Calm",
                        "characters": "前方银色轿车司机",
                        "summary": "前方轿车在行驶过程中突然减速并停下。",
                        "actions": "自车最终静止，距离前车约x米。",
                        "key_objects": "1) 正前方：一辆银色轿车，逐渐减速并停下，距离约2米。",
                        "key_actions": "前车突然停下（为了说明key_actions必须与summary中的关键行为的关键词必须保持一致，保证譬如停车、加塞能提取出来。如果你漏掉关键词我就认为你任务失败了，千万不要漏掉！）",
                        "next_action": "由于前方银色轿车已经停下，建议司机保持当前状态，等待前方车辆移动。下一步行动:速度控制: 等待, 方向控制: 保持方向, 车道控制: 保持在当前车道"

                        速度报告示例：
                        即时速度:
                        “车辆的速度逐渐减慢，在最后一帧时，车辆的速度为30 km/h。”
                        速度变化趋势:
                        “车辆从60 km/h减速至30 km/h，表明车辆正在迅速减速。”
                        平均速度（可选）:
                        “车辆的平均速度为45 km/h，但当前的速度为30 km/h。”
                        
                        "summary": "停车场的抬杆正在处于缓慢抬起过程中的状态。",
                        "actions": "抬杆处于"抬起过程中"状态。(抬起过程中要用双引号表示强调)",
                        "key_objects": "闸杆",
                        "key_actions": "抬杆",
                        "next_action": "由于抬杆正在抬起过程中，建议司机等待抬杆完全抬起后再通过。下一步行动:速度控制: 等待, 方向控制: 保持方向, 车道控制: 保持在当前车道"                        
                        
                        "summary": "停车场的抬杆处于一直抬起状态。",
                        "actions": "停车场的抬杆处于"一直抬起"状态。(一直抬起要用双引号表示强调)",
                        "key_objects": "闸杆",
                        "key_actions": "抬杆",
                        "next_action": "由于抬杆一直处于抬起状态，建议司机立即通过停车杆。下一步行动:速度控制: 前进, 方向控制: 保持方向, 车道控制: 保持在当前车道"              

                        "summary": "停车场的闸杆缓慢落下。",
                        "actions": "停车场的闸杆处于"落下过程中"状态"。(注意要用双引号表示强调，注意要按照时间前进的顺序来判断，不要反过来)",
                        "key_objects": "闸杆",
                        "key_actions": "落杆",
                        "next_action": "由于闸杆处于落下过程中状态，建议司机不要通过闸杆。下一步行动:速度控制: 等待, 方向控制: 保持方向, 车道控制: 保持在当前车道"                                      

                        "summary": "停车场的闸杆一直关闭。",
                        "actions": "停车场的闸杆处于"一直关闭"的水平状态"。(注意要用双引号表示强调)",
                        "key_objects": "闸杆",
                        "key_actions": "无",
                        "next_action": "由于停车场的闸杆处于一直关闭状态，建议司机不要通过闸杆。下一步行动:速度控制: 等待, 方向控制: 保持方向, 车道控制: 保持在当前车道"      

                        
                        
                    # Few-shot learning metadata
                    Below are time series example images and their corresponding analysis to help you understand how to analyze and label the images:
                    {fsl_payload} -> {assistant_response_relativeposition}
                    {lowerbar_fsl_payload} -> {assistant_response_lowerbar}
                    {raisebar_fsl_payload} -> {assistant_response_raisebar}
                            
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
AnalyzeVideo("test_video/car-driving.mov",4,10)

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