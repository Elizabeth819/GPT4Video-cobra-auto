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

#Audio API type (OpenAI, Azure)* c
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
    frame_interval = fi  # Chunk video evenly into segments by a certain interval, unit: seconds 
    frames_per_interval = fpi # Number of frames to capture per interval, unit: frames
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
        url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2024-08-01-preview" #2024-06-01
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
        json_form=str([json.dumps({"Start_Timestamp":"4.97s","sentiment":"Positive, Negative, or Neutral","End_Timestamp":"16s","scene_theme":"Dramatic","characters":"Characters is an array containing all the characters detected in the current scene.For each character, always provide the seven fields:'is_child','number_of_chilren','current_child','gender','location','wearing_seat_belt','Sleeping'.Example:[Man in hat, woman in jacket,{'is_child':'Yes','number_of_children':2,'current_child':'1','gender':'Male','location':'Rearleft'(distinguish each kid by location),'wearing_seat_belt':'否','Sleeping':'是,因为眼睛闭着,'description':'A boy about 6years old,wearing a blueT-shirt and jeans,sitting in the rear left seat without a seatbelt.'}]","summary":"Summary of what is occuring around this timestamp with actions included, uses both transcript and frames to create full picture, be detailed and attentive, be serious and straightforward in your description.Focus strongly on seatbelt location on the body and actions related to seatbelts.","actions":"Actions extracted via frame analysis","key_objects":"Any objects in the timerange, include colors along with descriptions. all people should be in this, with as much detail as possible extracted from the frame (clothing,colors,age) Be incredibly detailed","key_actions":"action labels extracted from actions, but do not miss any key actions listed in tasks","prediction":"Prediction of what will happen next especially seatbelt related actions or dangerous actions, based on the current scene."}),
        json.dumps({"Start_Timestamp":"16s","sentiment":"Positive, Negative, or Neutral","End_Timestamp":"120s","scene_theme":"Emotional, Heartfelt","characters":"Man in hat, woman in jacket","summary":"Summary of what is occuring around this timestamp with actions included, uses both transcript and frames to create full picture, detailed and attentive, be serious and straightforward in your description.","actions":"Actions extracted via frame analysis","key_objects":"Any objects in the timerange, include colors along with descriptions. all people should be in this, all people should be in this, with as much detail as possible extracted from the frame (clothing,colors,age). Be incredibly detailed","key_actions":"only key action labels extracted from actions","prediction":"Prediction of what will happen next, based on the current scene."})])
        
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
                    Make sure **EVERY** field in above json_form is filled out!!!
                    Do not make up timestamps, use the ones provided with each frame. 
                    Construct each action summary block from mulitple frames, each block should represent a scene or distinct trick or move in the video, minimum of 2 blocks per output.
                    Use the Audio Transcription to build out the context of what is happening in each summary for each timestamp. 
                    Consider all frames image by image and audio given to you to build the Action Summary. Be as descriptive and detailed as possible, 
                    Make sure to try and **Analyze the frames image by image** as a cohesive 10 seconds of video.
                    
                    ---
                    ***Tasks:***
                    You are an in-car AI video assistant to help passengers create comfortable environment. 
                    Task 1: Always execute and report in 'Characters' Recognition - PRIORITY ONE: 
                        - If there are children in the image, determine the following:
                            •Is it a child: Yes, No, Infant, Child
                            •Number of children: 0, 1, 2, 3...仔细观察可能被遮挡的位置,如果有人仅露出头发,大部分头部被遮挡,请把儿童数量也估算在内
                                    **RE-READING 儿童数量 IS REQUIRED FOR ALL TASKS**
                            •Gender: Male, Female, Unknown
                            •Location: Front passenger seat, Rear left, Rear center, Rear right
                            •Wearing seat belt: Yes, No, Unknown (if unsure, report it as Unknown)
                            •Sleeping: Yes(with criteria), No(with criteria)

                    Task 2: Children DANGEROUS Behavior Monitoring in Cabin - PRIORITY ONE:
                    **key_actions**: 
                      - Report *SPECIFIC* behaviors **TERMS** in **key_actions**,with a strong focus on critical behaviors include sticking their head out of the window, etc.
                      - Report specific behaviors in **key_actions**,with a strong focus on seat-belt related actions. Pay special attention to behaviors such as unbuckling the seat belt, attempting to escape from the seatbelt, or tampering with it.
                      - Extract the exact related actions into key_actions, eg, "挣脱安全带", not only report the final state of seat belt. If the state of seat belt has changed, you should report the specific action of changing the state of seat belt!!!
                        eg, key_actions: "孩子用如何动作导致挣脱了安全带" instead of "未系安全带".
                        1. If you find any child's head rested against the window, you should report children's head rested against which window of the car("头部靠着车窗").
                        2. If you find any childs head rested against the door, you should report children's head rested against which door of the car.
                        3. **HIGH PRIORITY ** If you find any childs **HEAD** *STICKING OUT OF WINDOW*, you should report children's head sticking out of which window("头伸出窗外").
                        4. If you find any child hold the door handle, you should report children holding which door handle.
                        5. If you find any child's hand resting on the door handle, you should report children's hand resting on which door handle.
                        6. If you find any child sticking their hand out of the window, you should report children sticking their hand out of which window.
                        7. If you find any child body sticking out of the window, you should report children sticking their body out of which window.
                        8. **HIGH PRIORITY ** If you find any child unbuckling their seat belts, you should report children unbuckling seat belt with exact .

                    Task 3: Children Behavior Monitoring in Cabin - PRIORITY TWO:
                        - Report specific behaviors in **key_actions**,with a strong focus on seat-belt related actions. Pay special attention to behaviors such as unbuckling the seat belt, attempting to escape from the seatbelt, or tampering with it.
                        - Extract the exact related actions into key_actions, eg, "挣脱安全带", not only report the final state of seat belt. If the state of seat belt has changed, you should report the specific action of changing the state of seat belt!!!
                            1.If you find any child closing eyes, dozing off or sleeping on the seat, you should report children sleeping in key_actions. 
                                a) Closed Eyes: Check if the passenger's eyes are closed, as this is a common indicator of sleep.
                                b) Head Position: Observe the passenger's head posture. If the head is slightly tilted back or in a relaxed position, it may suggest that the person is sleeping.
                                c) Body Posture: Examine the body posture. If the arms are crossed in front of the chest and the body appears relaxed and motionless, it might indicate the person is asleep.
                            2.If you find any child singing, you should report children singing in key_actions.
                            3.If you find any child eating something, you should report children eating in key_actions.
                                a) Hand-to-Mouth Movement: Watch for the child bringing food or utensils to their mouth. If the hand is positioned near the mouth, and the child is chewing or swallowing, it indicates eating, NOT TO BE CONFUSED WITH TOYS.
                                b) If the item held by the child is small and easy to hold, and its size is appropriate for single or double-handed gripping, such items are usually snacks or small food items, NOT TO BE CONFUSED WITH TOYS.
                                c) If the item's packaging or shape resembles common snack packaging, such as small bags, stick shapes, bars, or blocks, it can be inferred that the child might be eating something, NOT TO BE CONFUSED WITH TOYS.
                                By observing these indicators, you can accurately determine if a child is eating, not playing with toys.
                            5.If you find any child drinking, you should report children drinking or drinking through a straw in key_actions.
                            6.If you find any child gesticulate wildly, you should report children gesticulate wildly in key_actions.
                            7.If you find any child beating someone or something, you should report children beating sommething in key_actions.
                            8.If you find any child throwing something, you should report children throwing in key_actions.
                            9. If you find any child fighting something, you should call function of report children fighting in key_actions.
                            10.If you find any child attempting to **struggle** or **break free from their seat belt**, you should report children "unfastening the sea tbelt" in key_actions.
                                - **Struggling to unfasten the seatbelt** includes the following specific actions:
                                    a) The child **pulling or tugging** at the seatbelt with visible effort.
                                    b) **Reaching and grasping** the seatbelt buckle or strap multiple times.
                                    c) **Pushing or kicking** against the seatbelt or seat in an attempt to free themselves.
                                    d) **Twisting or turning** their body in an unnatural way while pulling at the seatbelt.
                                    e) **Crying or showing distress** while interacting with the seatbelt, which might often accompanies struggling.
                                - **Do not** only report the state like "not wearing a seatbelt" or "unbuckled seatbelt"; instead, focus on the specific ongoing action of struggling to break free from the seatbelt.
                                - Ensure to report any such actions in **key_actions** with clear descriptions.

                            11. If you find any child standing up or half-kneeling(半跪), you should report children standing up or half-kneeling(半跪) in key_actions.
                                "半跪" refers to a posture of 半蹲半站立,坐在脚上, 不是传统上的半跪指的是一条腿跪着,另一条腿站着,也要注意区别于仅仅"坐着".
                            12. If you find any child jumping/bouncing, you should call function of repoorting children jumping in key_actions.
                            13. If you find any child crying, you should report children crying in key_actions.
                            14. If you find any child laughing, you should report children laughing in key_actions.
                            15. If you find any child taking off their clothes, you should report children taking off their clothes in key_actions.
                            16. If you find any child pulling off the blanket, quilt/duvet, you should report children pulling off the blanket in key_actions.
                                The child grabbed the blanket forcefully, swung it around, moved it off, and then let it go, causing the blanket and quilt to be pulled off from their body. It's not just about the child holding the blanket.
                            17. If you find any child speaking or talking, you should report children speaking in key_actions.
                            18. If you find any child moving freely, you should report children's exact activities in key_actions.
                          
                            Among the behaviors mentioned above, **SLEEPING**, **STANDING-UP**, **TAKING OFF CLOTHES/BLANKET**, or **SEATBELT-RELATED ACTIONS** (such as **breaking free from the seatbelt** or **unfastening the seatbelt**) are **PRIORITY** behaviors. When you detect these behaviors, you should **report them FIRST** because the driver can see or feel other behaviors associated with these key actions.
                    
                    Task 4: **Proactive** Prediction of High-Risk Behaviors in Children - Priority One:
                            Predict the high-risk behaviors that the child may exhibit in the upcoming period, with a special focus on actions related to the **SEATBELT**. Pay close attention to the following early warning signs:
                            •Seatbelt Position Abnormalities: Such as the seatbelt sliding downwards, becoming loose, or not being worn correctly.
                            •Touching or Fidgeting with the Seatbelt: If the child is trying to adjust, pull, or play with the seatbelt, it may indicate they are about to unbuckle or interfere with it.
                            •Changes in Body Posture: Such as leaning forward or twisting, which may suggest an attempt to escape the seat or stick their head out of the window.
                            Other early signs to observe include:
                            •Distracted or Restless Behavior: May indicate impending high-risk actions.
                            •Direction of Gaze: Frequently looking at the window, door, or seatbelt buckle.
                            Please fill in the "prediction" field of the JSON output with the predicted behaviors based on these early signs.

                        "Behavior_Rules": {{
                            "Primary_Tasks": [
                                {{
                                    "Category": "儿童舱内行为监测",
                                    "Description": "识别儿童是否睡觉、站立/半跪、脱衣/扯掉毯子。",
                                    "Priority": "高",
                                    "Behaviors": [
                                        "睡眠",
                                        "站立/半跪",
                                        "脱衣/扯掉毯子"
                                    ]
                                }},
                                {{
                                    "Category": "儿童危险行为检测及预警",
                                    "Description": "检测儿童是否进行危险行为，如头伸出车窗、手伸出车窗等。",
                                    "Priority": "高",
                                    "Behaviors": [
                                        "儿童头伸出窗外",
                                        "儿童手伸出车窗",
                                        "儿童解开安全带"
                                    ]
                                }},
                                {{
                                    "Category": "儿童舱内行为监测",
                                    "Description": "监测儿童的日常行为，如吃喝、哭泣、笑等。",
                                    "Priority": "中",
                                    "Behaviors": [
                                        "吃",
                                        "喝",
                                        "说话",
                                        "哭",
                                        "笑",
                                        "跳跃",
                                        "打架",
                                        "打闹",
                                        "扔东西"
                                    ]
                                }}
                            ]
                        }},
                    
                        **重要注意事项**: 
                        1.上述多项任务是同时检测的，当您发现儿童在执行多项行为时，您应该同时报告这些行为，以便司机能够及时采取行动。如果喝东西的同时在说话，您应该同时报告这两项行为。
                        2.如果某个动作或手持物品不明确，您应该按照最大可能的行为或物品进行回答。如手持饮料又靠近嘴边,就很可能是在吃或喝东西,应该回答吃或喝东西,而不要只是汇报"物品",甚至"玩具",也不要说"手拿物品"!!!
                        对动作和物体应避免使用抽象词汇,要按越具体越好的原则给出你的猜测,否则你会视为不合格的执行了一次失败的任务,我会看不懂你的输出!!!
                        !!!!!不要使用"物品"这样的不明确词汇!!!!!不要使用手里拿着一个"物品"这样的模糊词汇!!!!!应该说:"正在吃东西"或"正在喝饮料"!!!!!

                    你现在是一名中文助手。无论我问什么问题，你都必须只用中文回答。请不要使用任何其他语言。You must always and only answer totally in **Chinese** language!!! I can only read Chinese language. Ensure all parts of the JSON output, including **summaries**, **actions**, and **next_action**, **MUST BE IN CHINESE** If you answer ANY word in English, you are fired immediately! Translate English to Chinese if there is English.

                    Your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON. Totally respond in Chinese language. Translate English to Chinese if there is any English words.仅用中文回答,如果有英文单词，请翻译成中文。
                    
                    Example 1 - Correct:
                    "Start_Timestamp": "0.5s",
                    "sentiment": "Neutral",
                    "End_Timestamp": "5.0s",
                    "scene_theme": "日常",
                    "characters": [
                        {{
                            "is_child": "是",
                            "number_of_children": "2",
                            "current_child": "1",
                            "gender": "未知",
                            "location": "后排左侧-可以看到一个人（可能是儿童）的头部，从前排座椅后面露出(不要因为看不到全部脸部就不算做一个人数!!!!!)",
                            "wearing_seat_belt": "是",
                            "Sleeping": "否",
                            "description": "一个戴着黄色帽子的孩子，坐在后排左侧的儿童座椅上，系着安全带/正在用手解开安全带, 手里拿着白色饮料,正在喝饮料",
                        }}
                    ],
                    "summary": "在这段视频中，一个孩子坐在后排左侧的儿童座椅上，戴着黄色的帽子并系着安全带,逐渐睡着, 前排座位上有一成年人",
                    "actions": "孩子坐在儿童座椅上睡着了, 孩子戴着黄色的帽子, 孩子系着安全带/正在解开安全带,坐着逐渐睡着了, 手里拿着零食正在吃零食, 孩子正在用吸管喝饮料",
                    "key_objects": "",
                    "key_actions": "孩子系着安全带/孩子用脚蹬踹,最终挣脱安全带,孩子坐着睡着了,扯掉毯子,吃零食, 喝饮料",
                    "prediction": "孩子安全带已滑落在脚底位置,孩子将要挣脱掉安全带,因此可能会摔倒,请注意安全"

                    Example 2 - Correct:
                    "summary":"安全带位置滑落在孩子脚底,仅有一部分在孩子腿上",
                    "key_actions":"孩子用脚蹬安全带,安全带位置在孩子脚底,孩子将要挣脱掉安全带,因此可能会摔倒,请注意安全",

                    Example 2 - Wrong: 只说"手里拿着一个物品"太过模糊,应该说"正在吃零食"或"正在喝饮料"!!!!!
                    "characters": [
                        {{
                            "is_child": "是",
                            ...
                            "description": "一个坐在后排左侧的孩子，穿着绿色的衣服，手里拿着一个物品(错误)。"
                        }},
                        {{
                            "is_child": "是",
                            ...
                            "description": "一个坐在后排右侧的孩子，手里拿着一个物品(错误)。"
                        }}
                    ],
                    "summary": "在这段视频中，两个孩子都在手里拿着物品(错误)。",
                    "actions": "两个孩子坐在后排座位上，手里拿着物品(错误)。",
                    "key_actions": "孩子们坐在后排座位上，手里拿着物品(错误)。"
                    "prediction": "-"
                    """
                    },
                {
                    "role": "user",
                    "content": cont
                }
                ],
                "max_tokens": 4000,
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
                    Your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON. Answer in Chinese language."""
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
                # convert string to json
                data=json.loads(vault, strict=False) #If strict is False (True is the default), then control characters will be allowed inside strings.Control characters in this context are those with character codes in the 0-31 range, including '\t' (tab), '\n', '\r' and '\0'.
                if isinstance(data, list):
                    data = data[0]

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
    with open('actionSummary.json', 'w', encoding='utf-8') as f:
    # Write the data to the file in JSON format
        json.dump(final_arr, f, indent=4, ensure_ascii=False)
        
    with open('transcript.txt', 'w') as f:
    # Write the data to the file in JSON format
        f.write(global_transcript)
    return final_arr

    #print("\n\n\n"+totalData)

#AnalyzeVideo("./medal.mp4",60,10)
# AnalyzeVideo("test_video/car-driving.mov",6,10,False)
# AnalyzeVideo("2024.9.1_2/瞌睡.mp4",5,10,False)
# AnalyzeVideo("2024.9.1/儿童低头睡觉.mp4",5,10,False)
AnalyzeVideo("2024.9.1/儿童身体伸出窗外-已伸出.mp4",5,5,False)
# 儿童车内脱衣服
# 毯子被扯掉
# 儿童半跪
# 安全带解开-从坐下到躺下
# 儿童低头睡觉
# 儿童挣脱安全带
# 儿童头伸向窗-未出去
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
