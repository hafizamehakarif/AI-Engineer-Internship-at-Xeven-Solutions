from fastapi import FastAPI

app = FastAPI()

#read data 
@app.get("/")

async def root():
    return {"message": "Hello World"}

#create data 
@app.post("/")

async def home():
    return{"message":"home"}

@app.post("/user")

async def create_user():
    return {"message":"User created"}

#delete data 
@app.delete("/user/{id}")

async def delete_user(id:int):
    return{"message":f"user {id} deleted"}