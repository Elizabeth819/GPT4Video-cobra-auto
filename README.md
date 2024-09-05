Hi everyone, general information and format for each of the .env file is below. (backend and frontend)

## Important Notes
  - Use GPT-4o instead of GPT-4-turbo vision for latest video interpretation capability.
  - The Azure GPT4 Vision service has 2 issues, 1: you can only send 10 (now 20, but unstable) images per call, so max FPI is 10, and you need to apply to turn of content filtering, as it is synchronous and adds 30+ seconds to each call.

## Frameworks and Languages:
  - Python 3.12 for the backend
  - NodeJS 18 and greater for the frontend, uses NextJS

## Installation Instructions:
  - Pull down the Github
  - set the ENV variables
  - (Install CMake beforehand: on Mac, brew install ffmpeg, brew install cmake; Linux ubuntu: sudo apt-get install cmake， and then install libgl1 for OpenGL: sudo apt install libgl1)
  - Pip install the requirements.txt in the backend 
  - npm install the frontend packages in the COBRA folder

## Run Instructions Backend:
  - Check if frames and audio folder exist, if existing, delete them, they are intermediate processing frames and extracted audios during running below ActionSummary-predict_explain-fsl-sys-Cn-cutin.py or ActionSummary.py.
  - For autonomous driving video analysis, run the ActionSummary-predict_explain-fsl-sys-Cn-cutin.py, video file name can be changed in the last line before main entry at the bottom.
  - For general video analysis, run the ActionSummary.py file from terminal with command like "python ActionSummary.py "path_to_video","frame interval integer","frames per interval integer","False". For example python ActionSummary.py "./myVideo.mp4",10,10,False.
  - For ChapterAnalysis, in the code, change the video path and fi/fpi directly at the end of the script.

## Run Instructions Frontend:
  - Linux shell: sudo apt install npm; npm install
  - navigate to the COBRA folder and "npm run dev" to start the dev server
  - if there is a nextjs error, delete the .next folder from the file directory.
  - to run in production npm run build, and then npm start: npm run dev
  - If using an Azure VM, add an inbound policy in settings, network, destination port 3000|TCP|Allow|priority=100
  - Open in local web browser: localhost:3000, or connect to a remote VM: your-vm-public-ip:3000

## Run Vector Search in Frontend searchbox:
  - Run ingest_scenes.py to create index and ingest actionSummary.json to AI Search
    change json file name, index name, ai search endpoint/api key in ingest_scenes.py
  - Modify app/api/cog/route.js: in last few lines, change semantic configuration name to the same as the semantic config used in the index creation json.
  - Modify index name in Cobra env file.
  - Search in the searchbox in a semantic way, get semantically similar results without having to exact match with keyword.

## Containers:
  - There are Docker Containers ready to go for both frontend and backend, as well as a compose that allows for facial recognition.

## Env Template Backend
  - AZURE_SPEECH_KEY= "This is not neccesary, it is for an eventual azure speech integration"
  - AZURE_WHISPER_KEY= "This is the key to your whisper instance in azure"
  - AZURE_WHISPER_DEPLOYMENT="deployment name of you azure whisper"
  - AZURE_WHISPER_ENDPOINT="Full endpoint url of your azure whisper 'https://jfjafjajf'"
  - AZURE_VISION_KEY="Your Azure OpenAI key for GPT4V"
  - AUDIO_API_TYPE="Azure or OpenAI"
  - VISION_API_TYPE="Azure or OpenAI"
  - OPENAI_API_KEY="OpenAI API key"
  - VISION_DEPLOYMENT_NAME="Azure OpenAI GPT4V deployment name"
  - VISION_ENDPOINT="Just endpoint name for Azure GPT4V 'ovv-vision'"

## Env Template Frontend
  - AZ_OPENAI_KEY="Azure OpenAI key"
  - AZ_OPENAI_REG=canadaeast
  - AZ_OPENAI_BASE=https://{your endpoint name}.openai.azure.com/
  - AZ_OPENAI_VERSION=2024-02-01
  - GPT4=4turbo
  - SEARCH_ENDPOINT=https://{your_search_service).search.windows.net
  - SEARCH_API_KEY="Your Search API Key"
  - INDEX_NAME=Your Index Name

## Enhancements:
  - Using florence to pre-process the frames as embeddings so we can optimally determine the best FPI for any given FI, making the solutional always optimal on depth
  - Full UI experience with the ability to upload and process video
  - Better Azure Whisper/Speech support.
  - Proper audio timestamping and diarization
  - better ENV strategy
  - integration with response_format when not preview.

## Priority Use cases:
  - Sequential action detection, explanation and prediction
  - Key actions and extensive objects identification
  - Audio dubbing for the visually disabled with accurate text insertion and neural voice
  - Dynamic ad insertion via semantic insertion
  - Multi video episodic analysis
  - End credit ID
  - Character Tracking/ID
  - video content moderation
  - anomaly detection
  - emotive sentiment analysis
  

  
