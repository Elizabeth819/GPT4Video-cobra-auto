'use client';
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { CardTitle, CardDescription, CardHeader, CardContent, Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { DropdownMenuTrigger, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuItem, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { CollapsibleTrigger, CollapsibleContent, Collapsible } from "@/components/ui/collapsible"
import { DialogTrigger, DialogTitle, DialogHeader, DialogContent, Dialog } from "@/components/ui/dialog"
import { useChat } from 'ai/react'
import axios from "axios"
import { useState, useRef,useEffect,useMemo } from "react"
import ReactPlayer from 'react-player';  
import actionSummary from '../../app/data/actionSummary.json'
import chapters from '../../app/data/chapterBreakdown.json'
import Image from "next/image";
// import Chat from './chat';

const SceneCard = ({ scene }) => {  
  return (  
    <Collapsible className="space-y-2">  
      <div className="flex items-center justify-between space-x-4 px-4">  
        <h4 className="text-sm font-semibold">{scene.title}</h4>  
        <CollapsibleTrigger asChild>  
          <Button size="sm" variant="ghost">  
            T<span className="sr-only">Toggle</span>  
          </Button>  
        </CollapsibleTrigger>  
      </div>  
      <CollapsibleContent className="space-y-2 px-4">  
        <p className="text-sm">{scene.description}</p>  
      </CollapsibleContent>  
    </Collapsible>  
  );  
};  
  
const ChapterCard = ({ chapterData }) => {  
  return (  
    <Card>  
      <CardHeader>  
        <CardTitle>Chapter Analysis</CardTitle>  
      </CardHeader>  
      <CardContent>  
        {Object.keys(chapterData).map((chapterKey, index) => {  
          const chapter = chapterData[chapterKey];  
          return (  
            <Collapsible key={index} className="space-y-2">  
              <div className="flex items-center justify-between space-x-4 px-4">  
                <h4 className="text-sm font-semibold">{chapter.title}</h4>  
                <CollapsibleTrigger asChild>  
                  <Button size="sm" variant="ghost">  
                    T<span className="sr-only">Toggle</span>  
                  </Button>  
                </CollapsibleTrigger>  
              </div>  
              <CollapsibleContent className="space-y-2">  
                {chapter.scenes.map((scene, sceneIndex) => (  
                  <SceneCard key={sceneIndex} scene={scene} />  
                ))}  
              </CollapsibleContent>  
            </Collapsible>  
          );  
        })}  
      </CardContent>  
    </Card>  
  );  
};  
export function Chat() {
  
  const { messages, input, handleInputChange, handleSubmit,setMessages,append } = useChat()
  let sysDefault="You are COBRA, an AI Video Analyzer. You are provided a chapter summary of the video in question, along with key scenes related to the users query. Use both to answer the user's question as faithfully and comprehensively as possible. be as succinct as possible while still conveying all key information. Use the chapter summary as the basis for your response, dont reference it by name or directly."
  sysDefault+="\nChapter Breakdown: \n"+JSON.stringify(chapters)
  const messDefault= {"role":"system","id":"0","createdAt":new Date(),"content":sysDefault}
  function clearChat(){
    setMessages([messDefault])
  }
  useState(()=>{
    setMessages([messDefault])
  },[])
  return (
    (<Dialog>
      <DialogTrigger asChild>
        <Button
          className="text-gray-800 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700"
          variant="outline">
          Open AI Chatbot
        </Button>
      </DialogTrigger>
      <DialogContent
        className="w-full max-w-xl h-full max-h-[600px] flex flex-col bg-white dark:bg-gray-900 shadow-xl rounded-xl">
        <DialogHeader
          className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <DialogTitle className="text-gray-900 dark:text-white font-bold">AI Chatbot</DialogTitle>
          <div>
            <Button
              className="text-gray-500 dark:text-gray-200 hover:text-gray-700 dark:hover:text-white"
              size="icon"
              title="Close the dialog"
              variant="ghost">
              <PanelTopCloseIcon className="h-5 w-5" />
            </Button>
          </div>
        </DialogHeader>
        <ScrollArea className="flex-1 overflow-y-auto">
          <div className="space-y-4 p-4">
            {messages.map((message, index) => (
              <>
              {message.role === "assistant" ? (
                <div
              className="group flex flex-col gap-2 py-2 border border-gray-200 border-gray-300 rounded-lg shadow-sm bg-white dark:border-gray-700 dark:bg-gray-800 dark:border-gray-800">
              <div
                className="flex-1 whitespace-pre-wrap p-2 text-sm prose prose-sm prose-neutral dark:text-white dark:prose-invert">
                <p>
                  <span className="font-semibold">AI:</span>
                  {message.content}
                </p>
                <p
                  className="text-right text-xs tracking-wide text-gray-800 dark:text-gray-300">
                  June 26, 2024 15:16 PM
                </p>
              </div>
            </div>

              ):message.role==="user"?(
                <div
              className="group flex flex-col gap-2 py-2 border border-gray-200 border-gray-300 rounded-lg shadow-sm bg-white dark:border-gray-700 dark:bg-gray-800 dark:border-gray-800">
              <div
                className="flex-1 whitespace-pre-wrap p-2 text-sm prose prose-sm prose-neutral dark:text-white dark:prose-invert">
                <p>
                  <span className="font-semibold">Alice:</span>
                  {message.content}
                </p>
                <p
                  className="text-right text-xs tracking-wide text-gray-800 dark:text-gray-300">
                  Apr 22, 2024 10:15 AM
                </p>
              </div>
            </div>
              ):(<></>)}
              </>
            ))}
            
            
          </div>
        </ScrollArea>
        <form onSubmit={handleSubmit}>
        <div
        
        
        
          className="flex items-center reversed:flex-row-reverse gap-2 p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          
          <Input
            className="bending-transition"
            clearable
            placeholder="Ask me anything!"
            type="text"
            onChange={handleInputChange}
            />
            
          <Button
            className="text-scheme-accent bg-transparent hover:bg-gray-200 dark:hover:bg-gray-700"
            variant="outline">
            <PlusIcon className="h-5 w-5" />
            <span className="sr-only">Add attachment</span>
          </Button>
          <Button onClick={handleSubmit}
            className="text-white bg-blue-500 hover:bg-blue-600 dark:hover:bg-blue-700">
            <SendIcon className="h-5 w-5 mr-2" />
            <span>Send</span>
            
          </Button>
          
        </div>
        </form>
      </DialogContent>
    </Dialog>)
  );
}

function PanelTopCloseIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
      <line x1="3" x2="21" y1="9" y2="9" />
      <path d="m9 16 3-3 3 3" />
    </svg>)
  );
}


function PlusIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <path d="M5 12h14" />
      <path d="M12 5v14" />
    </svg>)
  );
}


function SendIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <path d="m22 2-7 20-4-9-9-4Z" />
      <path d="M22 2 11 13" />
    </svg>)
  );
}

export function Player(props) {
  const [searchText, setSearchText] = useState('');   
  const [searchResults, setSearchResults] = useState([]); 
  const [played, setPlayed] = useState(0);  
  const [seeking, setSeeking] = useState(false);  
  const [playing, setPlaying] = useState(true);  
  const [modalIsOpen, setIsOpen] = useState(false);  
  const [currentChapter, setCurrentChapter] = useState(null);  
  const player = useRef(null);  
  const colors = ['primary', 'secondary', 'success', 'danger', 'warning', 'info', 'dark'];
  // const [videoURL,setVideoURL] = useState('./Conserve.mp4')
  // const [videoURL,setVideoURL] = useState('./car-driving.mov')
    // const [videoURL,setVideoURL] = useState('./鬼探头.mp4')
    // const [videoURL,setVideoURL] = useState('./guitantou.mp4')
    const [videoURL,setVideoURL] = useState('./复杂场景.mov')
  const videoSummaries = actionSummary
  
const parseTimestamp = (timestamp) => {  
  return parseFloat(timestamp.replace('s', ''));  
};  

const handleSearchResultClick = (startFrame) => {  
  setPlaying(true);  
  setTimeout(() => {  
    player.current.seekTo(startFrame);  
  }, 1000);  
}; 

const VideoSearch=({res})=>{
  const startFrame = parseInt(res.Start_Timestamp);  
  const endFrame = parseInt(res.End_Timestamp); 
  return(
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        {/* <Image
          alt="Thumbnail"
          className="rounded-md object-cover aspect-video"
          height={30}
          src="/placeholder.svg"
          width={40} /> */}
        <span className="text-sm line-clamp-1">{res.Start_Timestamp}</span>  
        <span className="text-sm line-clamp-1">{res.End_Timestamp}</span>  
        <span className="text-sm line-clamp-1">{res.summary.substring(0,40)}</span>
      </div>
      <PlayIcon className="w-4 h-4" onClick={()=>handleSearchResultClick(startFrame)}/>
    </div>
  )
}
  
const VideoSummaryDisplay = ({ currentSecond }) => {  
  const currentSummary = useMemo(() => {  
    const currentSummaryData = videoSummaries.find(summary => {  
      const start = parseTimestamp(summary.Start_Timestamp);  
      const end = parseTimestamp(summary.End_Timestamp);  
      return currentSecond >= start && currentSecond <= end;  
    });  
    return currentSummaryData ? currentSummaryData.summary : '';  
  }, [currentSecond]);  
  
  return (  
    <Textarea className="w-full h-52" value={currentSummary} readOnly />  
  );  
};  
  const [sentimentPercentages, setSentimentPercentages] = useState(null)

  const [duration, setDuration] = useState(0);  
  function calculateSentimentPercentages(data) {  
    const sentimentCounts = data.reduce((acc, item) => {  
      acc[item.sentiment] = (acc[item.sentiment] || 0) + 1;  
      return acc;  
    }, {});  
    
    const total = data.length;  
    const sentimentPercentages = {  
      Positive: ((sentimentCounts.Positive || 0) / total * 100).toFixed(2) + '%',  
      Negative: ((sentimentCounts.Negative || 0) / total * 100).toFixed(2) + '%',  
      Neutral: ((sentimentCounts.Neutral || 0) / total * 100).toFixed(2) + '%',  
    };  
    
    return sentimentPercentages;  
  }  
    
  useEffect(()=>{
    setSentimentPercentages(calculateSentimentPercentages(actionSummary))
  },[])
    
  const sentimentDistribution = calculateSentimentPercentages(actionSummary);  
  
  const handleDuration = (duration) => {  
    setDuration(duration);  
  };  
  // const handleProgress = ({ played }) => {  
  //   if (!seeking) {  
  //     setPlayed(played);  
  //   }  
  // };  
  const [currentSecond, setCurrentSecond] = useState(0);
  const handleProgress = ({ playedSeconds }) => {  
    setCurrentSecond(playedSeconds);  
  };  
  const handleSeekMouseDown = () => {  
    setSeeking(true);  
  };  
  
  const handleSeekChange = (e) => {  
    setPlayed(parseFloat(e.target.value));  
  };  
  
  const handleSeekMouseUp = (e) => {  
    setSeeking(false);  
    player.current.seekTo(parseFloat(e.target.value));  
  };  
  const handleSearchSubmit = (e) => {  
    e.preventDefault();  
    axios.post('/api/cog', { messages: searchText })  
      .then((response) => {  
        console.log(response)
        let count=0
        const results = response.data.message.map(item => {  
          count+=1
          return(
            {  
              ...item,  
              start_frame: item.Start_Timestamp.slice(0, -1),  
              end_frame: item.End_Timestamp.slice(0, -1) ,
              rank:count 
            }
          )
        });  
        // const results = response.data.message.map(item => ({  
        //   ...item,  
        //   start_frame: item.Start_Timestamp.slice(0, -1),  
        //   end_frame: item.End_Timestamp.slice(0, -1) ,
        //   rank:count 
        // }));  
        setSearchResults(results);  
      })  
      .catch((error) => {  
        console.error(error);  
      });  
  };  
  return (
    (<div
      key="1"
      className="grid min-h-screen w-full lg:grid-cols-[280px_1fr]">
      <div className="hidden border-r bg-gray-100/40 lg:block dark:bg-gray-800/40">
        <div className="flex h-full max-h-screen flex-col gap-2">
          <div className="flex h-[60px] items-center border-b px-6">
            <Link className="flex items-center gap-2 font-semibold" href="#">
              <AppWindowIcon className="h-6 w-6" />
              <span className="">COBRA</span>
            </Link>
            <Button className="ml-auto h-8 w-8" size="icon" variant="outline">
              <BellIcon className="h-4 w-4" />
              <span className="sr-only">Toggle notifications</span>
            </Button>
          </div>
          <div className="flex-1 overflow-auto py-2">
            <nav className="grid items-start px-4 text-sm font-medium">
              <Link
                className="group flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                href="#">
                <PlayIcon className="h-4 w-4" />
                Recent Videos
              </Link>
              <Link
                className="group flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                href="#">
                <FolderIcon className="h-4 w-4" />
                Collections
              </Link>
              <Link
                className="group flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                href="#">
                <SearchIcon className="h-4 w-4" />
                Search
              </Link>
              <Link
                className="group flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                href="#">
                <UploadIcon className="h-4 w-4" />
                Upload Video
              </Link>
            </nav>
          </div>
          <div className="mt-auto p-4">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle>Chat with Video</CardTitle>
                <CardDescription>Get real-time analysis and insights on your videos.</CardDescription>
              </CardHeader>
              <CardContent>
                <Chat />
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
      <div className="flex flex-col">
        <header
          className="flex h-14 lg:h-[60px] items-center gap-4 border-b bg-gray-100/40 px-6 dark:bg-gray-800/40">
          <Link className="lg:hidden" href="#">
            <AppWindowIcon className="h-6 w-6" />
            <span className="sr-only">Home</span>
          </Link>
          <div className="w-full flex-1">
            <form>
              <div className="relative">
                <SearchIcon
                  className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500 dark:text-gray-400" />
                <Input
                  className="w-full bg-white shadow-none appearance-none pl-8 md:w-2/3 lg:w-1/3 dark:bg-gray-950"
                  placeholder="Search video..."
                  type="search" 
                  onChange={(e) => setSearchText(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearchSubmit(e)}
                  />
              </div>
            </form>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                className="rounded-full border border-gray-200 w-8 h-8 dark:border-gray-800 dark:border-gray-800"
                size="icon"
                variant="ghost">
                <Image
                  alt="Avatar"
                  className="rounded-full"
                  height="32"
                  src="/placeholder.svg"
                  style={{
                    aspectRatio: "32/32",
                    objectFit: "cover",
                  }}
                  width="32" />
                <span className="sr-only">Toggle user menu</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Settings</DropdownMenuItem>
              <DropdownMenuItem>Support</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Logout</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </header>
        <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
          <div className="flex w-full gap-4">
            <div className="w-full rounded-lg overflow-hidden">
            <ReactPlayer    
              ref={player}    
              url={videoURL}    
              playing={playing}    
              controls={true}
              onProgress={handleProgress}    
              onDuration={handleDuration}  
              className="w-full aspect-video rounded-md bg-gray-100 dark:bg-gray-800" 
              width={"100%"}
              height={"70%"}
            /> 
            
            </div>
            <div className="grid w-[300px] gap-2">
              <Card className="h-4/5">
                <CardHeader>
                  <CardTitle>Sentiment Analysis</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-2">
                  <div className="flex items-center gap-2">
                    <SmileIcon className="w-5 h-5 text-green-500" />
                    <span>Positive: {sentimentDistribution?sentimentDistribution.Positive:"0%"}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <FrownIcon className="w-5 h-5 text-red-500" />
                    <span>Negative: {sentimentDistribution?sentimentDistribution.Negative:"0%"}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <MehIcon className="w-5 h-5 text-yellow-500" />
                    <span>Neutral: {sentimentDistribution?sentimentDistribution.Neutral:"0%"}</span>
                  </div>
                </CardContent>
              </Card>
              <Card className="h-4/5">
                <CardHeader>
                  <CardTitle>Topic Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm">
                    A video displayed the driver's point of view on an urban city road.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 h-64" style={{marginTop:"-5rem"}}>
            <Card>
              <CardHeader>
                <CardTitle>Matched Searches</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-2">
                {searchResults.map((result,inddex)=>(
                  <VideoSearch key={inddex} res={result}/>
                ))}
                
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Action Summary</CardTitle>
              </CardHeader>
              <CardContent>
              <VideoSummaryDisplay currentSecond={currentSecond}/>
              </CardContent>
            </Card>
            <ChapterCard chapterData={chapters} />
          </div>
        </main>
      </div>
      {/* <div className="hidden lg:block">
        <div className="grid grid-cols-1 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Transcript</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[250px]">
                <div className="p-2 text-sm">
                  <p>Today, we're introducing the frontend cloud...</p>
                  <p>Where frontend developers build, test, and deploy high-quality web applications...</p>
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
        <div className="grid grid-cols-1">
          <Card>
            <CardHeader>
              <CardTitle>Chapter Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <Collapsible className="space-y-2">
                <div className="flex items-center justify-between space-x-4 px-4">
                  <h4 className="text-sm font-semibold">Chapter 1</h4>
                  <CollapsibleTrigger asChild>
                    <Button size="sm" variant="ghost">
                      T
                      <span className="sr-only">Toggle</span>
                    </Button>
                  </CollapsibleTrigger>
                </div>
                <CollapsibleContent className="space-y-2">
                  <div
                    className="rounded-md border border-gray-200 px-4 py-2 font-mono text-sm shadow-sm dark:border-gray-800 dark:border-gray-800">
                    Scene 1: Introduction
                  </div>
                  <div
                    className="rounded-md border border-gray-200 px-4 py-2 font-mono text-sm shadow-sm dark:border-gray-800 dark:border-gray-800">
                    Scene 2: Problem Statement
                  </div>
                </CollapsibleContent>
              </Collapsible>
              <Collapsible className="space-y-2">
                <div className="flex items-center justify-between space-x-4 px-4">
                  <h4 className="text-sm font-semibold">Chapter 2</h4>
                  <CollapsibleTrigger asChild>
                    <Button size="sm" variant="ghost">
                      T
                      <span className="sr-only">Toggle</span>
                    </Button>
                  </CollapsibleTrigger>
                </div>
                <CollapsibleContent className="space-y-2">
                  <div
                    className="rounded-md border border-gray-200 px-4 py-2 font-mono text-sm shadow-sm dark:border-gray-800 dark:border-gray-800">
                    Scene 1: Solution Overview
                  </div>
                  <div
                    className="rounded-md border border-gray-200 px-4 py-2 font-mono text-sm shadow-sm dark:border-gray-800 dark:border-gray-800">
                    Scene 2: Technical Details
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </CardContent>
          </Card>
        </div>
      </div> */}
    </div>)
  );
}

function AppWindowIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <path d="M10 4v4" />
      <path d="M2 8h20" />
      <path d="M6 4v4" />
    </svg>)
  );
}


function BellIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
      <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
    </svg>)
  );
}


function FolderIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <path
        d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z" />
    </svg>)
  );
}


function FrownIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M16 16s-1.5-2-4-2-4 2-4 2" />
      <line x1="9" x2="9.01" y1="9" y2="9" />
      <line x1="15" x2="15.01" y1="9" y2="9" />
    </svg>)
  );
}


function MehIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="8" x2="16" y1="15" y2="15" />
      <line x1="9" x2="9.01" y1="9" y2="9" />
      <line x1="15" x2="15.01" y1="9" y2="9" />
    </svg>)
  );
}


function PlayIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>)
  );
}


function SearchIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>)
  );
}


function SmileIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M8 14s1.5 2 4 2 4-2 4-2" />
      <line x1="9" x2="9.01" y1="9" y2="9" />
      <line x1="15" x2="15.01" y1="9" y2="9" />
    </svg>)
  );
}


function UploadIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>)
  );
}
