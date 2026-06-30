from typing import List
from pydantic import BaseModel, Field
from typing import Optional

class RecommendRequest(BaseModel):
    user_id: Optional[str] = None
    history: List[str] = Field(default_factory = list)

class NewsItem(BaseModel):
    news_id: str
    title: str
    category: str
    score: float

class RecommendResponse(BaseModel):
    user_id: str
    cold_start: bool                 
    trending: List[NewsItem]       
    cbf: List[NewsItem]       
    item_cf: List[NewsItem]       
    mf_cf: List[NewsItem]       
