
# FASTAPI BASICS - Your First FastAPI Application

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# 1. Create FastAPI app
app = FastAPI(
    title="My First FastAPI App",
    description="Learn REST API basics",
    version="1.0.0"
)

# 2. Pydantic models (for data validation)
class ItemCreate(BaseModel):
    """What we expect when creating an item"""
    name: str
    price: float
    description: Optional[str] = None

class ItemResponse(BaseModel):
    """What we return after creating an item"""
    id: int
    name: str
    price: float
    description: Optional[str] = None

# 3. Simple database (in-memory)
items_db = {
    1: ItemResponse(id=1, name="Laptop", price=120000, description="Gaming laptop"),
    2: ItemResponse(id=2, name="Phone", price=80000, description="Smartphone"),
}

# ENDPOINT 1: GET /health
@app.get("/health")
def health_check():
    """Check if server is running"""
    return {
        "status": "healthy",
        "message": "✅ FastAPI is running!",
        "teacher": "Mehak's Tutorial"
    }

# ENDPOINT 2: GET /items/{id} (PATH PARAMETER)

@app.get("/items/{item_id}")
def get_item(item_id: int):
    """Get one item by ID (item_id is in the URL path)"""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return items_db[item_id]


# ENDPOINT 3: GET /items (LIST ALL)

@app.get("/items")
def get_all_items():
    """Get all items"""
    return list(items_db.values())


# ENDPOINT 4: GET /items/search (QUERY PARAMETER)

@app.get("/items/search")
def search_items(search: Optional[str] = None):
    """Search items by name (search is after ? in URL)"""
    if not search:
        return list(items_db.values())
    
    results = [
        item for item in items_db.values()
        if search.lower() in item.name.lower()
    ]
    
    if not results:
        return JSONResponse(
            status_code=404,
            content={"detail": "No items found"}
        )
    return results

# ENDPOINT 5: POST /items (CREATE NEW)

@app.post("/items")
def create_item(item: ItemCreate):
    """Create a new item (data sent in request body as JSON)"""
    new_id = max(items_db.keys()) + 1
    new_item = ItemResponse(
        id=new_id,
        name=item.name,
        price=item.price,
        description=item.description
    )
    items_db[new_id] = new_item
    return new_item


# RUN THE SERVER
cd
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_basics:app", host="127.0.0.1", port=8000, reload=True)