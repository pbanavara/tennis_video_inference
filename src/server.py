from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
from fastapi.concurrency import run_in_threadpool
import aiofiles
import asyncio
import os
from capture import AnalyzeVideo
import cv2

class Video(BaseModel):
    name: str


app = FastAPI()

@app.get("/")
def root():
    return {"Hello world"}

def process_video(video_file_name):
    avd = AnalyzeVideo()
    cap = cv2.VideoCapture(video_file_name)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            model = avd.choose_model(True)
            yield avd.overlay_poses(frame, model)
        else:
            break

        cap.release()

@app.post("/async_video")
async def post_video(file: UploadFile = File(...)):
    try:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
            try:
                contents = await file.read()
                await temp.write(contents)
            except Exception:
                return {"message": "There was an error uploading the file"}
            finally:
                await file.close()
        
        res = await run_in_threadpool(process_video, temp.name)  # Pass temp.name to VideoCapture()
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        #os.remove(temp.name)
        print("Done")

    return res

@app.post("/video")
def detect_faces(file: UploadFile = File(...)):
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
        return {"message": e.with_traceback()}
    finally:
        #temp.close()  # the `with` statement above takes care of closing the file
        #os.remove(temp.name)
        print("Done")
        
    return {"Done uploading" : temp.name}

@app.post("/process_video")
def process_video(video: Video):
    avd = AnalyzeVideo()
    return StreamingResponse(avd.analyze_video(video.name, 
                                      pose_flag=True, web_flag=True),
                    media_type = 'multipart/x-mixed-replace; boundary=frame')


