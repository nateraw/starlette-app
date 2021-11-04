import functools
import logging
import os
from typing import Dict, Type, Any, Tuple

# from api_inference_community.routes import pipeline_route, status_ok
from app.pipelines import Pipeline, ImageClassificationPipeline
from app.pipelines.image_classification import ImageClassificationPipeline
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette.responses import JSONResponse, Response
from starlette.requests import Request
import time
from io import BytesIO
from PIL import Image

TASK = os.getenv("TASK")
MODEL_ID = os.getenv("MODEL_ID")
HF_HEADER_COMPUTE_TIME = "x-compute-time"
HF_HEADER_COMPUTE_TYPE = "x-compute-type"
HF_HEADER_COMPUTE_CHARACTERS = "x-compute-characters"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "cpu")


logger = logging.getLogger(__name__)


ALLOWED_TASKS: Dict[str, Type[Pipeline]] = {
    'image-classification': ImageClassificationPipeline
}


def normalize_payload(bpayload, task: str) -> Tuple[Any, Dict]:
    img = Image.open(BytesIO(bpayload))
    return img, {}


async def status_ok(request):
    return JSONResponse({"ok": "ok"})


async def pipeline_route(request: Request) -> Response:
    start = time.time()
    payload = await request.body()
    task = os.environ["TASK"]

    try:
        pipe = request.app.get_pipeline()
        inputs, params = normalize_payload(payload, task)
    except (EnvironmentError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return call_pipe(pipe, inputs, params, start)


def call_pipe(pipe: Any, inputs, params: Dict, start: float):
    status_code = 200
    try:
        outputs = pipe(inputs)
    except (AssertionError, ValueError) as e:
        outputs = {"error": str(e)}
        status_code = 400
    except Exception as e:
        outputs = {"error": "unknown error"}
        status_code = 500
        logger.error(f"There was an inference error: {e}")

    headers = {
        HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start),
        HF_HEADER_COMPUTE_TYPE: COMPUTE_TYPE,
        # https://stackoverflow.com/questions/43344819/reading-response-headers-with-fetch-api/44816592#44816592
        "access-control-expose-headers": f"{HF_HEADER_COMPUTE_TYPE}, {HF_HEADER_COMPUTE_TIME}",
    }

    return JSONResponse(
        outputs,
        headers=headers,
        status_code=status_code,
    )


@functools.lru_cache()
def get_pipeline() -> Pipeline:
    task = os.environ["TASK"]
    model_id = os.environ["MODEL_ID"]
    if task not in ALLOWED_TASKS:
        raise EnvironmentError(f"{task} is not a valid pipeline for model : {model_id}")
    return ALLOWED_TASKS[task](model_id)


routes = [
    Route("/{whatever:path}", status_ok),
    Route("/{whatever:path}", pipeline_route, methods=["POST"]),
]

middleware = [Middleware(GZipMiddleware, minimum_size=1000)]
if os.environ.get("DEBUG", "") == "1":
    from starlette.middleware.cors import CORSMiddleware

    middleware.append(
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_headers=["*"],
            allow_methods=["*"],
        )
    )

app = Starlette(routes=routes, middleware=middleware)


@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.handlers = [handler]

    # Link between `api-inference-community` and framework code.
    app.get_pipeline = get_pipeline
    try:
        get_pipeline()
    except Exception as e:
        # We can fail so we can show exception later.
        # raise RuntimeError("UGH!!")
        print(str(e))
        raise RuntimeError(str(e))


if __name__ == "__main__":
    try:
        get_pipeline()
    except Exception:
        # We can fail so we can show exception later.
        raise RuntimeError("UGH from __main__!")
