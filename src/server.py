from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
from fastapi.concurrency import run_in_threadpool
import aiofiles
import asyncio
import os
from capture import AnalyzeVideo
import cv2
from fastapi.middleware.cors import CORSMiddleware
import common
import time
import base64
import traceback

class Video(BaseModel):
    name: str


app = FastAPI()
origins = [
    "http://localhost:3000",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"Hello world"}
    

@app.post("/video")
async def upload_video(email: str = Form(...), file: UploadFile = File(...)):
    temp = NamedTemporaryFile("wb", dir="/tmp", delete=False)
    print(temp.name)
    try:
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents)
        except Exception as e:
            return {"message" : e.with_traceback()}
        finally:
            file.file.close()
        
    except Exception as e:
        return {"email": email, "error": e.with_traceback()}
    finally:
        #temp.close()  # the `with` statement above takes care of closing the file
        #os.remove(temp.name)
        print("Done writing input file")
    try:
        email_time = email + str(int(time.time()))
        out_file_name = base64.b64encode(email_time.encode())[:10].decode("utf-8") + ".mp4"
        avd = AnalyzeVideo()
        ret = avd.analyze_video(temp.name, pose_flag=True, web_flag=True, output_file_name=out_file_name)
        print("Return value", ret)
        return {"email": email, "out_file" : out_file_name}
    except Exception as e:
        print(traceback.format_exc())
        return {"email": email, "error": common.FILE_PROCESS_FAIL }

@app.get("/fetchVideo")
def fetch_video(fileName: str):
    """Placeholder method for retrieving a streaming video

    Args:
        video_file (str): _description_

    Returns:
        _type_: _description_
    """
    avd = AnalyzeVideo()
    print("File name", fileName)
    return StreamingResponse(avd.analyze_video(fileName, 
                                      pose_flag=True, web_flag=True),
                    media_type = 'multipart/x-mixed-replace; boundary=frame')

@app.get("/fetchYTvideo")
def fetch_yt_video(url: str):
    """Placeholder method for retrieving a streaming video

    Args:
        fileName (str): _description_

    Returns:
        _type_: _description_
    """
    avd = AnalyzeVideo()
    return StreamingResponse(avd.get_youtube_video(url, 
                                      pose_flag=True, web_flag=True),
                    media_type = 'multipart/x-mixed-replace; boundary=frame')


