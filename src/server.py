from fastapi import FastAPI, File, UploadFile, Form, Response
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
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
import redis
import logging
import hashlib

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
logging.basicConfig(filename = "../log/callindra.log", encoding="utf-8", level=logging.DEBUG,
                    format='%(asctime)s %(message)s')
r = redis.Redis(host = "localhost", port=6379, db=0)

@app.get("/")
def root():
    logging.info("Test successful")
    return {"Hello world"}
    
def get_out_file_name(email, input_file):
    n_email = email + input_file
    hash = hashlib.new('SHA256', bytes(n_email, encoding="utf-8")).digest()
    b32_hash = base64.b32encode(hash).lower().rstrip(b'=').decode()[:10]
    out_file_name = b32_hash + ".webm"
    return out_file_name
    

def store_email_file(email, out_file_name):
    r.rpush(email, out_file_name)
    
@app.post("/video")
async def upload_video(email: str = Form(...), file: UploadFile = File(...)):
    temp = NamedTemporaryFile("wb", dir="/tmp", delete=False)
    if file.size > 20000000:
        raise HTTPException(status_code=413, detail="File size larger than 20MB")
    out_file_name = get_out_file_name(email, file.filename)
    if bytes(out_file_name, 'utf-8') in r.lrange(email, 0, -1):
        return {"email": email, "out_file" : out_file_name}
        
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
        logging.debug("Done writing input file", temp.name)
    try:
        
        avd = AnalyzeVideo()
        ret = avd.analyze_video(temp.name, pose_flag=True, web_flag=True, output_file_name=out_file_name)
        store_email_file(email, out_file_name)
        logging.debug("Return value", ret)
        return {"email": email, "out_file" : out_file_name}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail = "Internal server error")

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


