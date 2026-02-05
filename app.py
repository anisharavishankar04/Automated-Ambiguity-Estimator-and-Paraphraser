import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from exam_pipeline import ExamPipeline

app = FastAPI()
pipeline = ExamPipeline()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": None}
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...)
):
    # 1. Read bytes into a variable once
    file_bytes = await file.read()
    
    # 2. Safety Check: Print to your terminal to confirm the file isn't empty
    print(f"DEBUG: Received file '{file.filename}' - Size: {len(file_bytes)} bytes")
    
    if len(file_bytes) == 0:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "results": [], "error": "Uploaded file is empty."}
        )

    # 3. Run the pipeline
    try:
        results = pipeline.run(file_bytes)
    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        results = []

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": results}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
