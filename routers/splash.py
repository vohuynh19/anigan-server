from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_root():
    return {"Name": "I am Anigan server"}