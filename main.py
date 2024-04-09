# Importing necessary modules
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from output_parsers import review_intel_parser

# Loading environment variables from .env file
load_dotenv()


# Review template with prompts on inputs requesting answer from OpenAI.
# Creates string along with using correct formatting within other class
def product_review(review: str) -> str:
    product_review_template = """
    given this product review {review} I want you to breakdown the:
    1: sentiment
    2: emotion
    3: product name
    4: problem
    \n{format_instructions}
    """

    review_prompt_template = PromptTemplate(
        input_variables=["review"],
        template=product_review_template,
        partial_variables={
            "format_instructions": review_intel_parser.get_format_instructions()
        },
    )

    # Correct modeling to get right outputs prompting the AI
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=review_prompt_template)
    res = chain.invoke(input={"review": review})

    return res["text"]


if __name__ == "__main__":
    # Reviews
    positive_customer_review = """
    "I can't rave enough about this smart thermostat! Not only does it look sleek on my wall, 
    but it's also incredibly intuitive to use. The mobile app allows me to adjust the temperature from anywhere, 
    which is super convenient. Plus, the energy-saving features have noticeably reduced my utility bills. 
    Installation was a breeze, thanks to the clear instructions provided. Overall, 
    it's made managing my home's climate a breeze, and I highly recommend it to anyone looking to upgrade their 
    thermostat."
    """
    neutral_customer_review = """
    "These noise-canceling headphones do a decent job of blocking out ambient noise, 
    making them suitable for use in noisy environments like airplanes or busy offices. The sound quality is acceptable, 
    though not exceptional compared to some other headphones in this price range. Comfort-wise, 
    they're alright for short listening sessions, but I found them slightly uncomfortable during extended wear. 
    The battery life is decent, lasting me through a full day of use. Overall, 
    they're okay headphones, but they didn't particularly impress me."
    """
    negative_customer_review = """
    "I had high hopes for this fitness tracker, but unfortunately, it fell short of my expectations. 
    The step-counting feature is wildly inaccurate, often overestimating or underestimating my activity levels. 
    The heart rate monitor is similarly unreliable, giving inconsistent readings during workouts. 
    The companion app is clunky and prone to crashing, making it frustrating to track my fitness progress. 
    Additionally, the battery life is disappointing, requiring frequent recharging. Overall, 
    I'm disappointed with this purchase and wouldn't recommend it to others in search of a reliable fitness tracker."
    """

    # Obtaining AI responses for the reviews
    llm_response = product_review(positive_customer_review)
    llm_response2 = product_review(negative_customer_review)
    llm_response3 = product_review(neutral_customer_review)

    # Printing the AI responses
    print(llm_response)
    print(llm_response2)
    print(llm_response3)
