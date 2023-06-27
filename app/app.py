import logging

import uvicorn as uvicorn
from fastapi.applications import FastAPI

from app.routers import falcon

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(falcon.router)

if __name__ == '__main__':
    uvicorn.run("app:app",
                host="0.0.0.0",
                port=8080,
                reload=False,
                log_level=logging.INFO
                )
