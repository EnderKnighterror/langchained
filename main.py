from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from output_parsers import review_intel_parser


def product_review(review: str) -> str:
    product_review_template = """
    given this product review {review} I want you to breakdown the:
    1: sentiment
    2: emotion
    3: product name
    4: problem
    \n{format_instructions}
    """

    review_prompt_template = PromptTemplate(input_variables=["review"], template=product_review_template,
                                            partial_variables={
                                                "format_instructions": review_intel_parser.get_format_instructions()})

    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    chain = LLMChain(llm=llm, prompt=review_prompt_template)
    res = chain.invoke(input={"review": review})

    return res["text"]


if __name__ == "__main__":
    load_dotenv()

    customer_review = """
    Absolutely love my new wireless headphones! The sound quality is fantastic, and the Bluetooth 
    connectivity makes it so convenient. Plus, the battery life is impressive. Definitely worth the purchase!
    """

    llm_response = product_review(customer_review)
    print(llm_response)