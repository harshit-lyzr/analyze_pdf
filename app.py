import concurrent.futures
from fastapi import FastAPI, UploadFile, File, HTTPException
from llama_cloud_services import LlamaParse
from llamaparsing import get_combined_content
from mistralai import Mistral
from pathlib import Path
import shutil
from dotenv import load_dotenv
from llama_index.core.readers import SimpleDirectoryReader
from mistral_ocr import combine_markdown
import asyncio
import os
import time
from gpt_ocr import GPT4VisionClient, convert_pdf_to_images
from mistralai import DocumentURLChunk

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()
api_key = os.getenv("MISTRAL_KEY")
client = Mistral(api_key=api_key)
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")

@app.post("/analyze-pdf/")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF and get a combined analysis string.

    :param file: Uploaded PDF file.
    :return: Combined analysis string.
    """
    start = time.time()
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Save uploaded file temporarily
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as temp_file:
        temp_file.write(await file.read())

    # Convert PDF to images
    image_paths = convert_pdf_to_images(temp_pdf_path)

    # Remove the temporary PDF file
    os.remove(temp_pdf_path)
    extract_end = time.time()

    extract_final = extract_end - start
    # Initialize the GPT-4 Vision client  # Replace with your OpenAI API key
    client = GPT4VisionClient(openai_key)

    # Analyze each image and combine the outputs
    combined_output = ""
    prompt = "Extract all Detailes related to cliam from given image. Make it in structured manner."
    vision_start =time.time()

    def process_image(image_path, prompt):
        result = client.analyze_image(image_path, prompt)
        os.remove(image_path)  # Remove processed image
        return result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda path: process_image(path, prompt), image_paths)

    combined_output = "\n".join(results)
    vision_end = time.time()

    total_vision = vision_end - vision_start
    end = time.time()
    final = end - start
    return {"combined_output": combined_output,"total_time":final, "vision_time":total_vision,"extract_final":extract_final}


parser = LlamaParse(
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name=os.getenv("LLAMA_MODEL")
)


@app.post("/llama_ocr/")
async def parse_pdf(file: UploadFile = File(...)):
    file_location = f"/tmp/{file.filename}"

    # Save uploaded file
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Use SimpleDirectoryReader asynchronously
    file_extractor = {".pdf": parser}

    loop = asyncio.get_event_loop()
    documents = await loop.run_in_executor(None, lambda: SimpleDirectoryReader(
        input_files=[file_location], file_extractor=file_extractor
    ).load_data())

    combined_data = get_combined_content(documents)

    return {"parsed_text": str(combined_data)}


@app.post("/mistral_ocr/")
async def process_pdf(file: UploadFile = File(...)):
    try:
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pdf_file = Path(temp_file_path)
        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_file.stem,
                "content": pdf_file.read_bytes(),
            },
            purpose="ocr",
        )

        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True,
        )

        combined_text = combine_markdown(pdf_response)
        return {"combined_text": combined_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

