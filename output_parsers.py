# Importing necessary modules
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Defining the Pydantic model for review intelligence
class ReviewIntel(BaseModel):
    sentiment: str = Field(description="the sentiment of the review")
    emotion: str = Field(description="the emotion conveyed by the review")
    product_name: str = Field(description="name of the product")
    problem: str = Field(description="any problems identified in the review")

    # Custom method to convert the Pydantic model to a dictionary
    def to_dict(self):
        return {
            "sentiment": self.sentiment,
            "emotion": self.emotion,  # corrected "emotions" to "emotion"
            "product_name": self.product_name,
            "problems": self.problem,  # corrected "problems" to "problem"
        }


# Creating an instance of PydanticOutputParser with ReviewIntel model
review_intel_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=ReviewIntel
)
