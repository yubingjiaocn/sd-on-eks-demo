import os
import sys
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import boto3
import json
import uuid
from datetime import datetime
import httpx
import requests
import random
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

logger = logging.getLogger("demo-backend")
logger.propagate = False
logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Set current logger as global
logger = logging.getLogger("demo-backend")

app = FastAPI()

DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "sd-image-generation-tasks")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "arn:aws:sns:us-west-2:600413481647:sdoneksStack-sdoneksStacksdNotificationOutputCfn1C3DC221-pR2LpeipULCp")
SD_API_KEY = os.environ.get("SD_API_KEY", "bu2J4QVDxy9g3tP3aHWhW1NCvhCsya7j9yTMktRl")
SD_API_ENDPOINT = os.environ.get("SD_API_ENDPOINT", "https://e1lcc5lji3.execute-api.us-west-2.amazonaws.com/dev/v1alpha2")
CF_URL = os.environ.get("CF_URL", "https://d49x78c4g79qn.cloudfront.net/")

REQUEST_TEMPLATE = json.loads("""{
  "task": {
    "metadata": {
      "id": "",
      "runtime": "",
      "tasktype": "pipeline",
      "prefix": "output",
      "context": ""
    },
    "content": {}
  }
}""")

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

def load_template(template_name: str) -> str:
    with open(template_name, "r") as f:
        logger.info(f"Loading template from {template_name}")
        return f.read()

def generate_request(model: str, task_id: str, prompt: str, instance_type: str) -> dict:
    logger.info(f"Generating template for {task_id}")

    runtime = f"comfyui-{model}-{instance_type}"
    logger.info(f"Using {runtime} runtime")

    pipeline = load_template(f"templates/{model}.json")
    pipeline = pipeline.replace("$PROMPT", prompt)
    pipeline = pipeline.replace("$SEED", str(random.randint(0, 2147483648)))
    logger.debug(pipeline)

    merged = REQUEST_TEMPLATE.copy()
    merged["task"]["metadata"]["id"] = task_id
    merged["task"]["metadata"]["runtime"] = runtime
    merged["task"]["content"] = json.loads(pipeline)
    return merged

def generate_image(model: str, prompt: str, instance_type: str) -> str:
    task_id = str(uuid.uuid4())
    logger.info(f"Generating image for {task_id} with prompt: {prompt}")
    pipeline = json.dumps(generate_request(model, task_id, prompt, instance_type))
    logger.debug(pipeline)
    try:
        response = requests.post(url=SD_API_ENDPOINT, data=pipeline, headers={"x-api-key": SD_API_KEY, "Content-Type": "application/json"})
        logger.info(response.content.decode())
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    store_task(task_id)
    return task_id

def store_task(task_id: str):
    table.put_item(
        Item={
            'task_id': task_id,
            'status': 'submitted',
            'submit_time': datetime.now().isoformat(),
            'launch_time': "",
            'complete_time': "",
            'duration': "",
            'image_url': ""
        }
    )
    logger.info(f"Task {task_id} stored in DynamoDB")

def update_task_status(task_id: str, status: str, image_url: str = None):
    logger.info(f"Updating status for {task_id} to {status}")
    update_expression = "SET #status = :status"
    expression_attribute_names = {'#status': 'status'}
    expression_attribute_values = {
        ':status': status
    }

    if status == "running":
        update_expression += ", launch_time = :launch_time"
        expression_attribute_values[':launch_time'] = datetime.now().isoformat()

    if (status == "completed") & (image_url is not None):
        # get launch time from table
        task = get_task_status(task_id)

        launch_time = task.get("launch_time")
        complete_time = datetime.now().isoformat()
        duration = datetime.now() - datetime.fromisoformat(launch_time)

        image_cf_url = CF_URL + '/'.join(image_url.split("/")[3:])

        update_expression += ", image_url = :image_url, complete_time = :complete_time, process_duration = :process_duration"
        expression_attribute_values[':image_url'] = image_cf_url
        expression_attribute_values[':complete_time'] = complete_time
        expression_attribute_values[':process_duration'] = str(duration)

        logger.info(f"Task {task_id} completed with image URL: {image_cf_url}, use {duration}")

    table.update_item(
        Key={'task_id': task_id},
        UpdateExpression=update_expression,
        ExpressionAttributeNames=expression_attribute_names,
        ExpressionAttributeValues=expression_attribute_values
    )

def get_task_status(task_id: str) -> dict:
    response = table.get_item(Key={'task_id': task_id})
    return response.get('Item')

@app.post("/generate_with_same_instance_type")
async def generatee_with_same_instance_type(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    instance_type = data.get("instance_type")
    logger.info("Received new request")
    logger.debug(f"Prompt: {prompt}, Instance Type: {instance_type}")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if not instance_type:
        raise HTTPException(status_code=400, detail="Instance Type is required")
    logger.info("Sending SDXL image generation request")
    task_id_sdxl = generate_image("sdxl", prompt, instance_type)
    logger.info(f"Task ID for SDXL: {task_id_sdxl}")

    logger.info("Sending Flux.1 image generation request")
    task_id_flux = generate_image("flux", prompt, instance_type)
    logger.info(f"Task ID for Flux: {task_id_flux}")

    return {"task_id_sdxl": task_id_sdxl, "task_id_flux": task_id_flux}


@app.post("/generate_with_same_model")
async def generate_with_same_model(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    model = data.get("model")
    logger.info("Received new request")
    logger.debug(f"Prompt: {prompt}, Model: {model}")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if not model:
        raise HTTPException(status_code=400, detail="Model is required")

    logger.info("Sending image generation request to g5 instance")
    task_id_g5 = generate_image(model, prompt, "g5")
    logger.info(f"Task ID for g5: {task_id_g5}")

    logger.info("Sending image generation request to g6 instance")
    task_id_g6 = generate_image(model, prompt, "g6")
    logger.info(f"Task ID for g6: {task_id_g6}")

    logger.info("Sending image generation request to g6e instance")
    task_id_g6e = generate_image(model, prompt, "g6e")
    logger.info(f"Task ID for g6e: {task_id_g6e}")

    response = {
        "task_id_g5": task_id_g5,
        "task_id_g6": task_id_g6,
        "task_id_g6e": task_id_g6e,
    }

    return response


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    logger.info(f"Getting status for {task_id}")
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/sns-endpoint")
async def handle_sns_message(request: Request):
    message = await request.json()

    if message.get("Type") == "SubscriptionConfirmation":
        logger.info("Confirming subscription")
        subscription_url = message.get("SubscribeURL")
        async with httpx.AsyncClient() as client:
            await client.get(subscription_url)
        return JSONResponse(content={"message": "Subscription confirmed"}, status_code=200)

    if message.get("Type") == "Notification":
        logger.info("Processing notification")
        payload = json.loads(message.get("Message", "{}"))
        logger.debug(payload)
        task_id = payload.get("id")
        status = payload.get("status")
        if status == "running":
            update_task_status(task_id, status)

        if status == "completed":
            if payload.get("image_url"):
                update_task_status(task_id, status, payload.get("image_url")[0])

        if status == "failed":
            update_task_status(task_id, status)

        return JSONResponse(content={"message": "Notification processed"}, status_code=200)

    return JSONResponse(content={"message": "Unsupported message type"}, status_code=400)


@app.get("/")
async def health():
    return "healthy"