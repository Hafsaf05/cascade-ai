from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers.risk import router
from app.scoring import load_models


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    load_models()
    yield


app = FastAPI(title="Cascade - Advanced Risk Engineering for Minimum-Friction Risk Detection", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}