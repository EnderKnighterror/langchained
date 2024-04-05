from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ReviewIntel(BaseModel):
    sentiment: str = Field(description="the sentiment of the review")
    emotion: str = Field(description="the emotion conveyed by the review")
    product_name: str = Field(description="name of the product")
    problem: str = Field(description="any problems identified in the review")

    def to_dict(self):
        return {"sentiment": self.sentiment, "emotions": self.emotion, "product_name": self.product_name,
                "problems": self.problem}


review_intel_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=ReviewIntel)