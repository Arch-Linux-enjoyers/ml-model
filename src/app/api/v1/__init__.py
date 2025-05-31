from fastapi import APIRouter

from .endpoints import ml


api_router = APIRouter()
api_router.include_router(ml.router, prefix='/ml')