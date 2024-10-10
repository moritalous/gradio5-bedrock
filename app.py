import base64
import io
import json
import uuid
from mimetypes import guess_type

import boto3
import gradio as gr
from PIL import Image

model_id = "us.anthropic.claude-3-haiku-20240307-v1:0"


def converse_fn(message: str, history: list[gr.ChatMessage]):

    messages = []

    for h in history:
        messages.append({"role": h["role"], "content": [{"text": h["content"]}]})

    messages.append({"role": "user", "content": [{"text": message}]})

    client = boto3.client("bedrock-runtime")
    response = client.converse(modelId=model_id, messages=messages)

    return {
        "role": response["output"]["message"]["role"],
        "content": response["output"]["message"]["content"][0]["text"],
    }


def converse_streaming_fn(message: str, history: list[gr.ChatMessage]):

    messages = []

    for h in history:
        messages.append({"role": h["role"], "content": [{"text": h["content"]}]})

    messages.append({"role": "user", "content": [{"text": message}]})

    client = boto3.client("bedrock-runtime")
    response = client.converse_stream(modelId=model_id, messages=messages)

    stream = response["stream"]

    text = ""
    for chunk in stream:
        if "contentBlockDelta" in chunk:
            content_block_delta = chunk["contentBlockDelta"]
            text = text + content_block_delta["delta"]["text"]
            yield text


image_mimetype = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

document_mimetype = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
}


def multimodal_fn(message: str, history: list[gr.ChatMessage]):

    text = message["text"]
    files = message["files"]

    image_content = []
    document_content = []

    for file in files:
        with open(file, "rb") as f:
            mimetype, _ = guess_type(file)

            if mimetype in image_mimetype.keys():
                image_content.append(
                    {
                        "image": {
                            "format": image_mimetype[mimetype],
                            "source": {"bytes": f.read()},
                        }
                    }
                )
            if mimetype in document_mimetype:
                document_content.append(
                    {
                        "document": {
                            "format": document_mimetype[mimetype],
                            "name": str(uuid.uuid4())[:8],
                            "source": {"bytes": f.read()},
                        }
                    }
                )

    messages = []

    for h in history:
        # テキストのときだけ追加する。ファイルの場合はtupleになる
        if type(h["content"]) is str:
            messages.append({"role": h["role"], "content": [{"text": h["content"]}]})

    content = [content for content in image_content]
    content.append({"text": text})
    content.extend([content for content in document_content])

    messages.append(
        {
            "role": "user",
            "content": content,
        }
    )

    client = boto3.client("bedrock-runtime")
    response = client.converse_stream(modelId=model_id, messages=messages)

    stream = response["stream"]

    text = ""
    for chunk in stream:
        if "contentBlockDelta" in chunk:
            content_block_delta = chunk["contentBlockDelta"]
            text = text + content_block_delta["delta"]["text"]
            yield text


def image_fn(prompt: str):
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    response = client.invoke_model(
        modelId="stability.stable-image-ultra-v1:0", body=json.dumps({"prompt": prompt})
    )

    output_body = json.loads(response["body"].read().decode("utf-8"))
    base64_output_image = output_body["images"][0]
    image_data = base64.b64decode(base64_output_image)
    image = Image.open(io.BytesIO(image_data))

    return image


demo_converse = gr.ChatInterface(converse_fn, type="messages")
demo_converse_streaming = gr.ChatInterface(converse_streaming_fn, type="messages")
demo_multimodal = gr.ChatInterface(multimodal_fn, type="messages", multimodal=True)
demo_image = gr.Interface(image_fn, inputs=[gr.Text()], outputs=[gr.Image()])

demo = gr.TabbedInterface(
    [demo_converse, demo_converse_streaming, demo_multimodal, demo_image],
    ["Converse API", "Converse Stream API", "Multimodal", "Image"],
)

demo.launch()
